from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap, get_bev_gt
import torch.nn.functional as F
from lib.utils.grid_indexer import GridIndexer
import numpy as np
import lib.utils.geom as geom
from lib.utils import vox

class MITrackerMotActor(BaseActor):
    """ Actor for training MITracker models """
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight  # Loss weights for different loss components
        self.settings = settings  # Training settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg  # Configuration settings
        self.max_view_num = settings.max_view_num
        self.bevgt = GridIndexer([8000.0,8000.0,3000.0], [0,0,1450.0], [400,400,150])

        self.Y, self.Z, self.X = self.cfg.MODEL.PROJECT.RESOLUTION
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=torch.tensor(self.cfg.MODEL.PROJECT.SCENE_CENTROID).reshape([1, 3]),
            bounds=self.cfg.MODEL.PROJECT.BOUNDS, # [0, 400, 0, 400, 0, 3]  # xmin,xmax,ymin,ymax,zmin,zmax
            assert_cube=False)

        world_shape = self.cfg.MODEL.PROJECT.WORLD_SHAPE # [8,8,3] # m
        worldgrid_shape = [cfg.MODEL.PROJECT.BOUNDS[1], cfg.MODEL.PROJECT.BOUNDS[3], cfg.MODEL.PROJECT.BOUNDS[5]] 
        self.worldcoord_from_worldgrid_mat = np.array([[world_shape[0]/worldgrid_shape[0], 0, 0], 
                                                       [0, world_shape[1]/worldgrid_shape[1], 0], 
                                                       [0, 0, 1]]) 
        self.world_shape = world_shape
        self.worldgrid_shape = worldgrid_shape
        

    def __call__(self, data):
        # forward pass
        out_dict, augment = self.forward_pass(data) 
        # compute losses
        loss, status = self.compute_losses(out_dict, data, augment)  
        return loss, status  # Return the total loss and status dictionary

    def forward_pass(self, data):
        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldcoord_from_worldgrid[2, 2] = self.world_shape[2]/self.worldgrid_shape[2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid) 
        
        if self.net.training:
            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.08
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
        else:
            augment = torch.eye(4)

        out_dict = self.net(template=data['template_images'].reshape(self.max_view_num,-1, *data['template_images'].shape[1:]).transpose(0,1),
                            search=data['search_images'].reshape(self.max_view_num,-1, *data['search_images'].shape[1:]).transpose(0,1),
                            search_resize_transform=data['search_resize_transform'].reshape(self.max_view_num,-1, *data['search_resize_transform'].shape[1:]).transpose(0,1),
                            seq_view_id=data['seq_view_id'].transpose(0, 1),  
                            search_crop_bbox=data['search_crop_bbox'].reshape(self.max_view_num,-1, *data['search_crop_bbox'].shape[1:]).transpose(0,1), 
                            scene_names=data['scene_name'],
                            ref_T_global=worldgrid_T_worldcoord,
                            return_last_attn=False) 

        return out_dict, augment  # Return the output dictionary

    def compute_losses(self, pred_dict, gt_dict, augment, return_status=True):
        assert isinstance(pred_dict, list)  
        loss_dict = {}  
        total_status = {}  
        total_loss = torch.tensor(0., dtype=torch.float).cuda() 
        
        Ns = len(pred_dict)
        gt_dict['search_anno'] = gt_dict['search_anno'].reshape(self.max_view_num,-1, *gt_dict['search_anno'].shape[1:]).transpose(0,1).reshape(Ns,-1,4) 
        visible_masks = gt_dict['search_visible'].reshape(self.max_view_num,-1, *gt_dict['search_visible'].shape[1:]).transpose(0,1).reshape(Ns,-1)
 
        if 'score_map' in pred_dict[0] and self.loss_weight['score_map'] > 0:
            gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE) 
        if 'bev_heatmap' in pred_dict[0] and self.loss_weight['bev_heatmap'] > 0:
            worldgrid_pts_org = self.bevgt.compute_indices_2d(gt_dict['bev_xys']).clone().detach().type(torch.float32).reshape(-1,2)
            worldgrid_pts = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(1) 
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0).to(worldgrid_pts.device), worldgrid_pts)
            mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
            center_bev = get_bev_gt(mem_pts.squeeze(1), self.Y, self.X)
            gt_bev_maps_list = center_bev.view(Ns,-1,self.Y,self.X).to(gt_dict['bev_xys'].device) 
            
        for i in range(len(pred_dict)): 
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1) 
            
            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes'] 
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")  
            num_queries = pred_boxes.size(1) 
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0) 
            
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec, visible_masks[i]) 
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda() 
            loss_dict['giou'] = giou_loss  
            
            # compute l1 loss
            if visible_masks[i].sum() == 0:
                l1_loss = torch.tensor(0.0, device=giou_loss.device)
            else:
                l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec,reduction='none')
                l1_loss = l1_loss[visible_masks[i]!=0]
                l1_loss = l1_loss.mean()
            loss_dict['l1'] = l1_loss 
            
            # compute location loss
            if 'score_map' in pred_dict[i] and self.loss_weight['score_map'] > 0:
                location_loss = self.objective['score_map'](pred_dict[i]['score_map'], gt_gaussian_maps, visible_masks[i]) 
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device) 
            loss_dict['score_map'] = location_loss 

            if 'bev_heatmap' in pred_dict[i] and self.loss_weight['bev_heatmap'] > 0:
                bevmap_loss = self.objective['bev_heatmap'](pred_dict[i]['bev_heatmap'], gt_bev_maps_list[i])  
            else:
                bevmap_loss = torch.tensor(0.0, device=l1_loss.device)  
            loss_dict['bev_heatmap'] = bevmap_loss

            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)  # Compute weighted sum of losses
            total_loss += loss  # Accumulate total loss
            
            if return_status:
                # status for log
                status = {}  # Dictionary to store status for current frame
                
                mean_iou = iou.detach().mean()  # Compute mean IoU
                status = {f"{i}frame_Loss/total": loss.item(),
                        f"{i}frame_Loss/giou": giou_loss.item(),
                        f"{i}frame_Loss/l1": l1_loss.item(),
                        f"{i}frame_Loss/location": location_loss.item(),
                        f"{i}frame_IoU": mean_iou.item(),
                        f"{i}frame_Loss/bev_map": bevmap_loss.item(),
                        }  # Store loss and IoU values for logging
                
                total_status.update(status)  # Update total status

        if return_status:
            return total_loss, total_status  # Return total loss and status if requested
        else:
            return total_loss  # Return total loss only if status is not requested