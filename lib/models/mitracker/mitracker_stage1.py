import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.mitracker.vit_dinov2 import vit_large_patch14_518_dinov2, vit_base_patch14_518_dinov2
from lib.utils.box_ops import box_xyxy_to_cxcywh


class MITrackerStage1(nn.Module):
    """ This is the base class for MITracker model. """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz) # feature size of the search region
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"
        '''
        template: (numtemplate, B , 3, H, W) search: (numsearch, B, 3, H, W) for sot read
        template: (numtemplate, numview, B , 3, H, W) search: (numsearch, numview, B, 3, H, W) 
        '''
        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:] 
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2)) 
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()

            # Forward head
            out = self.forward_head(opt, None)
            out.update(aux_dict)
            out['backbone_feat'] = x 
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        bs, Nq, C, HW = opt.size() 
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


def build_mitracker_stage1(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
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

    box_head = build_box_head(cfg, hidden_dim)

    model = MITrackerStage1(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    return model