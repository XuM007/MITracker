from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap


class MITrackerSotActor(BaseActor):
    """ Actor for training MITracker Stage1 models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight  # Loss weights for different loss components
        self.settings = settings  # Training settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg  # Configuration settings

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)  # Perform a forward pass through the network

        # compute losses
        loss, status = self.compute_losses(out_dict, data)  # Compute losses based on predictions and ground truth

        return loss, status  # Return the total loss and status dictionary

    def forward_pass(self, data):
        template_list = []  
        search_list = [] 
    
        for i in range(self.settings.num_template): # N_t 
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  
            template_list.append(template_img_i)  

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  
            search_list.append(search_img_i) 
            
        out_dict = self.net(template=template_list,
                            search=search_list,
                            return_last_attn=False)  
        return out_dict  

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        assert isinstance(pred_dict, list) 
        loss_dict = {} 
        total_status = {}  
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  # Initialize total loss as a tensor on GPU
        
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)  
        
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
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda() 
            loss_dict['giou'] = giou_loss 
            
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec) 
            loss_dict['l1'] = l1_loss  
            
            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)  
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss  # Store focal loss
                
            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight) 
            total_loss += loss  # Accumulate total loss
            
            if return_status:
                status = {}  # Dictionary to store status for current frame
                
                mean_iou = iou.detach().mean()  # Compute mean IoU
                status = {f"{i}frame_Loss/total": loss.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_IoU": mean_iou.item()}  # Store loss and IoU values for logging
                
                total_status.update(status)  # Update total status

        if return_status:
            return total_loss, total_status  # Return total loss and status if requested
        else:
            return total_loss  # Return total loss only if status is not requested
