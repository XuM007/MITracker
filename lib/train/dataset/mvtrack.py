import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import json

class MVTrack_sot(BaseVideoDataset):
    """ MVTrack dataset.
        may change: multiview input, params.json load
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None): 
        root = env_settings().mvtrack_dir if root is None else root
        self.root = root
        super().__init__('mvtrack_sot', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)] 
        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            if split == 'train':
                file_path = os.path.join(self.root, 'train_split.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val_split.txt')
            elif split == 'test':
                file_path = os.path.join(self.root, 'test_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0][:-1]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]
        return seq_per_class

    def get_name(self):
        return 'mvtrack'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        invisible_file = os.path.join(seq_path, "invisible.txt")

        with open(invisible_file, 'r', newline='') as f:
            invisible = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~invisible
        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id] 
        class_name = seq_name.split('-')[0] 
        vid_id = seq_name.split('-')[1] 
        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) # if bbox shape (n,4), valid is (n,)
        visible = self._read_target_visible(seq_path) & valid.byte() # if bbox shape (n,4), visible is (n,)
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:05}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id) # different order like: ashbin1-1, pingpong4-2, ashbin1-3

        obj_class = self._get_class(seq_path)[:-1] 
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids] 

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

class MVTrack_mot(BaseVideoDataset):
    """ MVTrack dataset."""
    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None, 
                 sample_minview = 2): 
        root = env_settings().mvtrack_dir if root is None else root
        super().__init__('mvtrack_mot', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)] 
        self.calibpath = os.path.join(self.root, 'calibs.json')

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()
        self.seq2scene = self._get_scene_name(os.path.join(self.root, 'calibs.json'))
        self.sample_minview = sample_minview

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            if split == 'train':
                file_path = os.path.join(self.root, 'train_split.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val_split.txt')
            elif split == 'test':
                file_path = os.path.join(self.root, 'test_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0][:-1]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]
        return seq_per_class

    def get_name(self):
        return 'mvtrack'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]
    
    def _get_scene_name(self, calib_file):
        name = {}
        with open(calib_file, 'r') as f:
            calibration_version = json.load(f)['calibration_version']
        for scene, calib_version in calibration_version.items():
            for i in calib_version:
                name[i] = scene
        return name
            
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_bev_anno(self, seq_path):
        BEV_path = f'{seq_path}/../BEV/xyz_index.txt'
        xyz_index = pandas.read_csv(BEV_path, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        bev_xy = xyz_index[:, :2]
        cube_xyz = xyz_index[:,:3]
        indexs = xyz_index[:, 3]
        return torch.tensor(bev_xy), torch.tensor(cube_xyz), indexs
    
    def _read_cam_pose(self, seq_path):
        cam_params_file = os.path.join(seq_path, "params.json")
        with open(cam_params_file, 'r') as f:
            cam_params = json.load(f)
        extrinsic = np.array(cam_params['extrinsic'])
        return torch.tensor(extrinsic)
    
    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        invisible_file = os.path.join(seq_path, "invisible.txt") # occlusion or out of view

        with open(invisible_file, 'r', newline='') as f:
            invisible = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~invisible

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id] 
        class_name = seq_name.split('-')[0] 
        seq_num = len([st for st in os.listdir(os.path.join(self.root, class_name)) if st.startswith(class_name)])
        seqs_path_list = [os.path.join(self.root, class_name, class_name + '-' + str(ii)) for ii in range(1,seq_num+1) ]
        return random.sample(seqs_path_list,k=random.randint(self.sample_minview,seq_num))

    def get_sequence_info(self, seq_id): 
        seqs_path = self._get_sequence_path(seq_id) 
        seqs_info = []
        for seq_path in seqs_path:
            bbox = self._read_bb_anno(seq_path)
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) # if bbox shape (n,4), valid is (n,)
            visible = self._read_target_visible(seq_path) & valid.byte() 
            seqs_info.append({'bbox': bbox, 'valid': valid, 'visible': visible, 'seq_path': seq_path})
        return seqs_info # list

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:05}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno_list=None): 
        multiview_frame_list = None # [view1_s1, view1_s2, view2_s1, view2_s2]
        multiview_anno_frames = {}

        for anno in anno_list:
            seq_path = anno['seq_path']
            obj_class = self._get_class(seq_path)[:-1] 
            frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
            if multiview_frame_list is None:
                multiview_frame_list = frame_list
            else:
                multiview_frame_list += frame_list

            if anno is None:
                bbox = self._read_bb_anno(seq_path) #(n,4)
                valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) # if bbox shape (n,4), valid is (n,)
                visible = self._read_target_visible(seq_path) & valid.byte() 
                anno = {'bbox': bbox, 'valid': valid, 'visible': visible}

            for key, value in anno.items():
                if key == 'bbox':
                    frame_bbox = [value[f_id, ...].clone() for f_id in frame_ids]
                    if key not in multiview_anno_frames.keys():
                        multiview_anno_frames['bbox'] = frame_bbox
                    else:
                        multiview_anno_frames['bbox'] += frame_bbox

                elif key != 'seq_path':
                    if key not in multiview_anno_frames.keys():
                        multiview_anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
                    else:
                        multiview_anno_frames[key] += [value[f_id, ...].clone() for f_id in frame_ids]

            seq_view_id = int(seq_path.split('/')[-1][-1])
            if 'seq_view_id' not in multiview_anno_frames.keys():
                multiview_anno_frames['seq_view_id'] = [seq_view_id]
            else:
                multiview_anno_frames['seq_view_id'] += [seq_view_id]

        all_bev_xys, all_cube_xyz, indexs = self._read_bev_anno(anno['seq_path']) 
        bev_valid = True
        for f_id in frame_ids:
            if indexs[f_id] == -1:
                all_seq_name = [anno_list[i]['seq_path'].split('/')[-1] for i in range(len(anno_list))]
                bev_valid = False
        bev_xys = [all_bev_xys[f_id, ...].clone() for f_id in frame_ids]
        cube_xyz = [all_cube_xyz[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'scene_name': self.seq2scene[self.sequence_list[seq_id].split('-')[0]],
                                   'bev_xys': bev_xys,
                                   'cube_xyz': cube_xyz,
                                   'bev_valid': bev_valid,
                                   })
        return multiview_frame_list, multiview_anno_frames, object_meta