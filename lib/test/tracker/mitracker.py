import numpy as np
from lib.models.mitracker import build_mitracker
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import cv2
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box

class MITracker(BaseTracker):
    def __init__(self, params):
        super(MITracker, self).__init__(params)
        network = build_mitracker(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print('Load weights from %s' % self.params.checkpoint)
        self.cfg = params.cfg
        self.params = params
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        self.z_dict1 = {}

    def initialize(self, image_list, info_list: dict):
        self.z_patch_arr_dict = {}
        self.memory_frames_dict = {}
        self.memory_masks_dict = {}
        self.state_dict = {}
        if 'scene_name' in info_list[0]:
            self.scene_name = [info_list[0]['scene_name']]
        else:
            self.scene_name = None
        self.seq_view_ids = [[info['seq_view_id'] for info in info_list]] 
        # forward the template once
        for i, (image, info) in enumerate(zip(image_list, info_list)):
            z_patch_arr, resize_factor, z_amask_arr, crop_bbox, resize_transform = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            self.z_patch_arr_dict[i] = z_patch_arr
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)
            
            with torch.no_grad():
                self.memory_frames_dict[i] = [template.tensors]
                
            self.memory_masks_dict[i] = []
            self.state_dict[i] = info['init_bbox']
        self.frame_id = 0

    def track(self, image_list, info_list: dict = None):
        H, W, _ = image_list[0].shape
        self.frame_id += 1
        search_list = []
        template_list = []
        search_crop_bbox_list = []
        search_resize_transform_list = []
        resize_factor_list = []
        if self.frame_id > 1:
            self.state_dict = [info_list[i]['previous_output']['target_bbox'] for i in range(len(info_list))]
        for i, (image, info) in enumerate(zip(image_list, info_list)):
            x_patch_arr, resize_factor, x_amask_arr, crop_bbox, resize_transform = sample_target(image, self.state_dict[i], self.params.search_factor,
                                                                    output_sz=self.params.search_size)
            resize_factor_list.append(resize_factor)
            search_crop_bbox_list.append(crop_bbox)
            search_resize_transform_list.append(resize_transform)
            search_list.append(self.preprocessor.process(x_patch_arr, x_amask_arr).tensors)
            
            if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
                template_list.append(torch.stack(self.memory_frames_dict[i].copy(), dim=0).squeeze(0))
            else:
                template, box_mask_z = self.select_memory_frames(i)
                template_list.append(torch.stack(template, dim=0).squeeze(0))
                
        # # --------- select memory frames ---------
        search_in = torch.stack(search_list, dim=0).unsqueeze(0) # Nt, Nv, B, C, H, W
        template_in = torch.stack(template_list, dim=0).unsqueeze(0) # Ns, Nv, B, C, H, W
        search_resize_transform_in = torch.stack(search_resize_transform_list, dim=0).unsqueeze(0).unsqueeze(-3).to(search_in.device)
        search_crop_bbox_in = torch.stack(search_crop_bbox_list, dim=0).unsqueeze(0).unsqueeze(-2).to(search_in.device)
        with torch.no_grad():
            out_dict = self.network.forward(template=template_in, search=search_in, 
                                            search_resize_transform=search_resize_transform_in, search_crop_bbox=search_crop_bbox_in, 
                                            seq_view_id=torch.tensor(self.seq_view_ids).reshape(len(self.seq_view_ids),len(self.seq_view_ids[0])).to(search_in.device), 
                                            scene_names=self.scene_name)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map']) 
        pred_boxes = pred_boxes.view(-1, 4) # [Nv, 4]
        # Baseline: Take the mean of all pred boxes as the final result
        for box_i,(pred_box,image) in enumerate(zip(pred_boxes,image_list)):
            pred_box = (pred_box.unsqueeze(0).mean(dim=0) * self.params.search_size / resize_factor_list[box_i]).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state_dict[box_i] = clip_box(self.map_box_back(pred_box, resize_factor_list[box_i],self.state_dict[box_i]), H, W, margin=10)

            # --------- save memory frames and masks ---------
            z_patch_arr, z_resize_factor, z_amask_arr, crop_bbox, resize_transform = sample_target(image, self.state_dict[box_i], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
            frame = cur_frame.tensors
            if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
                frame = frame.detach().cpu()
            self.memory_frames_dict[box_i].append(frame)
        # --------- save memory frames and masks ---------
        
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        return [{"target_bbox": self.state_dict[i]} for i in range(len(image_list))]

    def select_memory_frames(self, view_i):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames_dict[view_i][idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            
        return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float, state):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float, state):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return MITracker
