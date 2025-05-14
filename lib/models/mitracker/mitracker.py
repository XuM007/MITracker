import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from typing import List
import numpy as np
import lib.utils.basic as basic
from lib.utils import vox
from lib.utils.cameras import adjust_camera_parameters, _get_cam
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.head import build_box_head, CenterPredictorBEV
from lib.models.layers.layernorm2d import LayerNorm2d
from lib.models.layers.bev import Res_Block
from lib.models.mitracker.vit_dinov2 import vit_large_patch14_518_dinov2, vit_base_patch14_518_dinov2
from timm.models.vision_transformer import Block as TransformerBlock

class MITracker(nn.Module):
    """ This is the base class for MITracker """
    def __init__(self, transformer, box_head, projectlayer=None, aux_loss=False, head_type="CORNER", training=True,
                 token_len=1, embed_dim=768, patch_size=14, feat_tp=None, max_view_num=None, heatmap_channel=32, cameras = None, 
                 resolution = None, world_shape = None, worldgrid_shape = None, z_sign=1, fixed_cam_num = False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.training = training
        # view specific feature extraction
        self.backbone = transformer
        self.box_head = box_head
        self.embed_dim = embed_dim
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz) 
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.feat_len_t = int(feat_tp ** 2)

        self.aux_loss = aux_loss
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.track_view_querys = [None for i in range(max_view_num)]
        self.cube_querys = None
        self.token_len = token_len
        
        # 3D feature projection and aggregation
        self.heatmap_channel = heatmap_channel
        self.world_shape = world_shape
        self.z_sign = z_sign
        self.fixed_cam_num = fixed_cam_num
        self.cameras = cameras
        self.Y, self.Z, self.X = resolution 
        worldcoord_from_worldgrid_mat = np.array([[world_shape[0]/worldgrid_shape[0], 0, 0], 
                                                    [0, world_shape[1]/worldgrid_shape[1], 0], 
                                                    [0, 0, 1]]) 
        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldcoord_from_worldgrid[2, 2] = world_shape[2]/worldgrid_shape[2]
        self.ref_T_global = torch.inverse(worldcoord_from_worldgrid)
        
        # layers
        self.resblock = Res_Block(heatmap_channel, embed_dim,layers=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transformer_block = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=8) for _ in range(3)])
        activation = nn.GELU
        self.upsample_enc = nn.Sequential(
            nn.ConvTranspose2d(embed_dim,  256, kernel_size=patch_size//2, stride=patch_size//2),
            LayerNorm2d(256),
            activation(),
            nn.ConvTranspose2d(256, self.heatmap_channel, kernel_size=2, stride=2),
            nn.ReLU(),
        ) 
        self.one_chan_heatmap = nn.Conv2d(heatmap_channel, 1, kernel_size=1, stride=1)
        
        self.project_layer = projectlayer
        self.softmax = nn.Softmax(dim=-1)
        self.bevhead = CenterPredictorBEV(inplanes=heatmap_channel, img_sz=self.X)
        self.cam_compressor = nn.Sequential(
            nn.Conv3d(heatmap_channel * max_view_num, heatmap_channel, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(heatmap_channel), nn.ReLU(),
            nn.Conv3d(heatmap_channel, heatmap_channel, kernel_size=1),
        )
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.heatmap_channel * self.Z, heatmap_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(heatmap_channel), nn.ReLU(),
            nn.Conv2d(heatmap_channel, heatmap_channel, kernel_size=1),
        )
        
    def forward(self, template: List[List[torch.Tensor]], 
                search: List[List[torch.Tensor]],
                search_resize_transform: List[List[torch.Tensor]],
                search_crop_bbox: List[List[torch.Tensor]],
                seq_view_id=None,
                scene_names=None,
                ref_T_global=None,
                return_last_attn=False,
                ):
        assert isinstance(search, torch.Tensor)
        '''
        template: (numtemplate, numview, B , 3, H, W) search: (numsearch, numview, B, 3, H, W)
        '''
        Ns, Nv, B, _, H, W = search.shape
        Nt = template.shape[0]

        enc_opt_all = []
        enc_tp_all = []
        featuremaps_sc_all = []
        for view_id in range(search.shape[1]): 
            for search_index in range(search.shape[0]):
                x, aux_dict = self.backbone(z=template[:,view_id], x=search[search_index][view_id],
                                        return_last_attn=return_last_attn, track_query=self.track_view_querys[view_id], token_len=self.token_len)
                feat_last = x 
                if isinstance(x, list):
                    feat_last = x[-1]
                
                enc_sc = feat_last[:, -self.feat_len_s:]  # (B, N_x, C)
                enc_tp = feat_last[:, -self.feat_len_s-self.feat_len_t*Nt:-self.feat_len_s] # (B, N_t, C)
                if self.backbone.add_cls_token:
                    self.track_query = (x[:, :self.token_len].clone()).detach() # (B, 1, C) 
                att = torch.matmul(enc_sc, x[:, :1].transpose(1, 2))  
                opt = (enc_sc.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous() # (B, 1, C, N_x)

                enc_opt_all.append(opt) # B, 1, C, N_x
                enc_tp = enc_tp.permute((0, 2, 1)).contiguous()
                enc_tp_all.append(enc_tp)
                enc_sc_patched = enc_sc.transpose(1,2).reshape(B, self.embed_dim, self.feat_sz_s, self.feat_sz_s) 
                featuremap_sc = self.upsample_enc(enc_sc_patched) 
                featuremaps_sc_all.append(featuremap_sc) 

        out_dict = []
        enc_opt_all = torch.stack(enc_opt_all, dim=0).reshape(Nv, Ns, *opt.shape).transpose(0, 1).squeeze(3).transpose(-1,-2) 
        enc_tp_all = torch.stack(enc_tp_all, dim=0).reshape(Nv, Ns, *enc_tp.shape).transpose(0, 1) 
        featuremaps_sc_all = torch.stack(featuremaps_sc_all, dim=0).reshape(Nv, Ns, *featuremap_sc.shape).transpose(0, 1) 
        out_dict = self.forward_vox(seq_view_id, featuremaps_sc_all, search_crop_bbox, search_resize_transform, scene_names, ref_T_global, enc_opt_all)
        return out_dict
    
    def forward_vox(self, seq_view_id, featuremaps_sc_all, search_crop_bbox, search_resize_transform, scene_names, ref_T_global, enc_opt_all):
        '''
        recall dimension
        enc_opt_all: Ns, Nv, B, 26*26, 768
        enc_tp_all: [Ns, Nv, B, 768, 13*13]
        featuremaps_sc_all: [Ns, Nv, B, 32, 364, 364]
        mean_enc_tp: [Ns, B, 768, 13*13]
        search_crop_bbox: [Ns, Nv, B, 4]
        search_resize_transform: [Ns, Nv, B, 2, 3]
        scene_names: [B]
        seq_view_id: [B, Nv]
        '''
        Ns, Nv, B = featuremaps_sc_all.shape[:3]
        if self.cameras is not None:
            if not self.training:
                ref_T_global = self.ref_T_global
            global_T_cams_, pixel_T_cams_ = self.cam_params(search_resize_transform, search_crop_bbox, seq_view_id, scene_names)
            ref_T_global_ = ref_T_global.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1).repeat(Ns, Nv, 1, 1).double()
            global_T_cams_ = global_T_cams_.reshape(Ns*Nv*B, 4, 4)
            ref_T_global_ = ref_T_global_.reshape(Ns*Nv*B, 4, 4).to(global_T_cams_.device)
            pixel_T_cams_ = pixel_T_cams_.reshape(Ns*Nv*B, 4, 4)
            ref_T_cams_ = torch.matmul(ref_T_global_, global_T_cams_) 
            cams_T_ref_ = torch.inverse(ref_T_cams_) 
            feat_cams_ = featuremaps_sc_all.reshape(Ns*Nv*B, *featuremaps_sc_all.shape[3:]).double() 
            featpix_T_cams_ = pixel_T_cams_

            # unproject image feature to 3d grid
            feat_mems = self.project_layer.unproject_image_to_mem(
                feat_cams_, 
                basic.matmul2(featpix_T_cams_, cams_T_ref_), 
                cams_T_ref_, self.Y, self.Z, self.X,
                xyz_refA=None, z_sign=self.z_sign) 
            
            # fuse multiview features
            feat_mems = feat_mems.reshape(Ns, Nv, B, *feat_mems.shape[1:]) 
            feat_mems = feat_mems.transpose(1,2) 
            feat_mems = feat_mems.reshape(Ns*B, Nv, *feat_mems.shape[3:]) 
            if self.fixed_cam_num:
                feat_mem = self.cam_compressor(feat_mems.flatten(1, 2))
            else:
                feat_mems = feat_mems.reshape(Ns, B, Nv, *feat_mems.shape[2:]) 
                for b in range(B):
                    for v in range(Nv):
                        view_id = int(seq_view_id[b, v])
                        if view_id == 0:
                            feat_mems[:, b, v] = torch.zeros_like(feat_mems[:, b, v])  # Assign all zeros
                feat_mems = feat_mems.reshape(Ns*B, Nv, *feat_mems.shape[3:]) 
                mask_mems = (torch.abs(feat_mems) > 0).float()
                feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1).float() 
            
            # feat_mem: Ns*B, latent_dim, Y, Z, X
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(Ns*B, self.heatmap_channel * self.Z, self.Y, self.X)
            feat_bev = self.bev_compressor(feat_bev_)
            bev_scoremap = self.bevhead(feat_bev) 
            bev_scoremap = bev_scoremap.reshape(Ns, B, *bev_scoremap.shape[2:]) 
            bev_token = self.resblock(feat_bev)
            bev_token = self.adaptive_pool(bev_token).reshape(Ns, B, self.embed_dim, 1).unsqueeze(2).repeat(1, 1, Nv, 1, 1).transpose(1,2) 
            bev_token = bev_token.reshape(Ns,Nv*B, 1, self.embed_dim) 

        out_dict = []
        if self.cameras is None:
            enc_opt_all = enc_opt_all.unsqueeze(-3).transpose(-1,-2)
        for search_idx in range(Ns):
            if self.cameras is not None:
                sctokens = enc_opt_all[search_idx].reshape(Nv*B, *enc_opt_all.shape[-2:])
                sctokens = torch.cat([bev_token[search_idx], sctokens], dim=1) 
                head_in = self.transformer_block(sctokens) 
                head_in = head_in[:, 1:].transpose(1,2).unsqueeze(1)      
            else:
                head_in = enc_opt_all[search_idx].reshape(-1, 1, self.embed_dim, self.feat_len_s)
            out = self.forward_head(head_in, None) 
            if self.cameras is not None:
                out['bev_heatmap'] = bev_scoremap[search_idx] 
                
            multi_chan_heatmap = featuremaps_sc_all[search_idx].reshape(Nv*B,*featuremaps_sc_all.shape[-3:]) 
            heatmap = self.one_chan_heatmap(multi_chan_heatmap).squeeze(-3).reshape(Nv*B,-1) 
            normalized_heatmaps = (heatmap - heatmap.min(-1, keepdim=True)[0]) / (heatmap.max(-1, keepdim=True)[0] - heatmap.min(-1, keepdim=True)[0] + 1e-6)
            normalized_heatmaps = normalized_heatmaps.view(Nv*B, *featuremaps_sc_all.shape[-2:])
            out['heatmap'] = normalized_heatmaps
            out_dict.append(out)
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        bs, Nq, C, HW = opt.size() # bs: batchsize, Nq:token_len, C, N_x
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
        
    def cam_params(self,search_resize_transform, search_crop_bbox, seq_view_id, scene_names):
        '''
        input:
            search_resize_transform: Ns, Nv, B, 2,3
            search_crop_bbox: Ns, Nv, B, 4
            seq_view_id: B, Nv
            scene_names: B
        output:
            cam->global: Ns, Nv, B, 4,4
            ref->global: Ns, Nv, B, 4,4
            pixel->cam: Ns, Nv, B, 4,4
        '''
        device = search_resize_transform.device
        global_T_cams_all = []
        pixel_T_cams_all = []
        Ns, Nv, B = search_crop_bbox.shape[:3]
        translation_shift = np.array([self.world_shape[0]/2, self.world_shape[1]/2, 0]).reshape(-1, 1)
        
        for search_idx in range(Ns):
            for c in range(Nv):
                for b in range(B):
                    view_id = int(seq_view_id[b, c])
                    if view_id == 0:
                        fix = torch.tensor([[1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[0, 0, 0, 1]], dtype=torch.double).to(device)
                        global_T_cams_all.append(fix)
                        pixel_T_cams_all.append(fix)
                        continue
                    scene = scene_names[b]
                    cam = self.cameras[scene][str(view_id)].copy()
                    crop_info = search_crop_bbox[search_idx, c, b]
                    resize_transform = search_resize_transform[search_idx, c, b]
                    pix_T_cams = adjust_camera_parameters(cam, crop_info, resize_transform)
                    pix_T_cams_ = torch.eye(4, dtype=torch.double)
                    pix_T_cams_[0,0] = pix_T_cams['fx']
                    pix_T_cams_[1,1] = pix_T_cams['fy']
                    pix_T_cams_[0,2] = pix_T_cams['cx']
                    pix_T_cams_[1,2] = pix_T_cams['cy']
                    pixel_T_cams_all.append(pix_T_cams_.to(device))

                    R = np.array(cam['R'])
                    T = np.array(cam['T']/1000).reshape(-1, 1)
                    T = T - R @ translation_shift 

                    R = R.T
                    T = -R @ T
                    global_T_cams = np.concatenate((R, T), axis=1) # world to camera
                    global_T_cams = np.concatenate((global_T_cams, np.array([[0, 0, 0, 1]])), axis=0)
                    global_T_cams = torch.tensor(global_T_cams, dtype=torch.double).to(device)
                    global_T_cams_all.append(global_T_cams)
        global_T_cams_ = torch.stack(global_T_cams_all, dim=0).reshape(Ns, Nv, B, 4, 4)
        pixel_T_cams_ = torch.stack(pixel_T_cams_all, dim=0).reshape(Ns, Nv, B, 4, 4)
        return global_T_cams_, pixel_T_cams_

def build_mitracker(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE!=" " and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        if not os.path.exists(pretrained):
            raise ValueError("Pretrained model not found at: {}".format(pretrained))
    else:
        pretrained = ''
        
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch14_518_dinov2':
        backbone = vit_base_patch14_518_dinov2(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                           )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch14_518_dinov2':
        backbone = vit_large_patch14_518_dinov2(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                            )
    else:
        raise NotImplementedError
    
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    
    if training and cfg.MODEL.PROJECT.CAMERAS != ' ':
        cameras = _get_cam([cfg.MODEL.PROJECT.CAMERA_PATH_MVTRACK],cfg.MODEL.PROJECT.CAMERAS)
        scene_centroid = torch.tensor(cfg.MODEL.PROJECT.SCENE_CENTROID).reshape([1, 3])
        Y, Z, X = cfg.MODEL.PROJECT.RESOLUTION
        bounds = cfg.MODEL.PROJECT.BOUNDS
        projectlayer = vox.VoxelUtil(Y, Z, X, scene_centroid=scene_centroid, bounds=bounds)
    elif not training and cfg.TEST.CAMERAS != ' ':
        cameras = _get_cam([cfg.TEST.CAMERA_PATH_MVTRACK],cfg.TEST.CAMERAS)
        scene_centroid = torch.tensor(cfg.TEST.SCENE_CENTROID).reshape([1, 3])
        Y, Z, X = cfg.TEST.RESOLUTION
        bounds = cfg.TEST.BOUNDS
        projectlayer = vox.VoxelUtil(Y, Z, X, scene_centroid=scene_centroid, bounds=bounds)
    else:
        cameras = None
        projectlayer = None
    
    box_head = build_box_head(cfg, hidden_dim)
    model = MITracker(
        backbone,
        box_head,
        projectlayer=projectlayer,
        head_type=cfg.MODEL.HEAD.TYPE,
        training=training,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        embed_dim=hidden_dim,
        patch_size=16 if '16' in cfg.MODEL.BACKBONE.TYPE else 14,
        feat_tp=int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.STRIDE),
        max_view_num=cfg.DATA.TRAIN.DATASETS_MAXVIEW,
        heatmap_channel=cfg.MODEL.BACKBONE.HEATMAP_CHANNEL,
        cameras=cameras,
        resolution=cfg.MODEL.PROJECT.RESOLUTION if training else cfg.TEST.RESOLUTION,
        world_shape=cfg.MODEL.PROJECT.WORLD_SHAPE if training else cfg.TEST.WORLD_SHAPE,
        worldgrid_shape=[cfg.MODEL.PROJECT.BOUNDS[1], cfg.MODEL.PROJECT.BOUNDS[3], cfg.MODEL.PROJECT.BOUNDS[5]] if training else [cfg.TEST.BOUNDS[1], cfg.TEST.BOUNDS[3], cfg.TEST.BOUNDS[5]],
        z_sign=cfg.MODEL.PROJECT.ZSIGN,
    )

    load_from = cfg.MODEL.PRETRAIN_PTH
    if load_from == " ":
        return model
    if training:
        layerstoload = cfg.MODEL.PRETRAIN_LAYER 
        checkpoint = torch.load(load_from, map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["net"].items() if k.split('.')[0] in layerstoload}
        model_dict.update(pretrained_dict)
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        print('Load pretrained model from: ' + load_from)
        
    if 'sequence' in cfg.MODEL.PRETRAIN_FILE and training:
        print("i change myself")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    return model