import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import pandas
import json

class MVTrackMotDataset(BaseDataset):
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.mvtrack_dir
        if split is not None:
            if split == 'train':
                self.file_path = os.path.join(self.base_path, 'train_split.txt')
            elif split == 'val':
                self.file_path = os.path.join(self.base_path, 'val_split.txt')
            elif split == 'test':
                self.file_path = os.path.join(self.base_path, 'test_split.txt')
            else:
                raise ValueError('Unknown split name.')
        else:
            raise ValueError('Set split_name.')
        
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()
        self.seq2scene = self._get_scene_name(os.path.join(self.base_path, 'calibs.json'))
    
    def _get_scene_name(self, calib_file):
        name = {}
        with open(calib_file, 'r') as f:
            calibration_version = json.load(f)['calibration_version']
        for scene, calib_version in calibration_version.items():
            for i in calib_version:
                name[i] = scene
        return name

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)): # ashbin1-1
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  set(clean_lst)

    def get_sequence_list(self):
        multi_view_list = []
        for name in self.clean_list:
            seq_list = [i for i in self.sequence_list if name in i]
            multi_view_list.append([self._construct_sequence(s) for s in seq_list])

        return multi_view_list

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0] # ashbin1
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        invisible_label_path = '{}/{}/{}/invisible.txt'.format(self.base_path, class_name, sequence_name)

        if os.path.exists(invisible_label_path):
            invisible = load_text(str(invisible_label_path), delimiter=',', dtype=np.float64, backend='numpy')
            target_visible = np.logical_not(invisible).astype(np.float32)
        else:
            target_visible = None

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name

        scene_name = self.seq2scene[class_name]
        seq_view_id = int(sequence_name.split('-')[1])
        return Sequence(sequence_name, frames_list, 'mvtrack_mot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible,
                        scene_name=scene_name, seq_view_id=seq_view_id)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = pandas.read_csv(self.file_path, header=None).squeeze("columns").values.tolist()
        return sequence_list


class MVTrackSotDataset(BaseDataset):
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.mvtrack_dir
        if split is not None:
            if split == 'train':
                self.file_path = os.path.join(self.base_path, 'train_split.txt')
            elif split == 'val':
                self.file_path = os.path.join(self.base_path, 'val_split.txt')
            elif split == 'test':
                self.file_path = os.path.join(self.base_path, 'test_split.txt')
            else:
                raise ValueError('Unknown split name.')
        else:
            raise ValueError('Set split_name.')
        
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)): # ashbin1-1
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls[:-1])
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0] # ashbin1
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        invisible_label_path = '{}/{}/{}/invisible.txt'.format(self.base_path, class_name, sequence_name)
        invisible = load_text(str(invisible_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        
        target_visible = np.logical_not(invisible).astype(np.float32)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'mvtrack_sot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = pandas.read_csv(self.file_path, header=None).squeeze("columns").values.tolist()
        return sequence_list
