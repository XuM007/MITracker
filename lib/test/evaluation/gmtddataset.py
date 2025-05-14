import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class GMTDSotDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gmtd_dir
        self.sequence_list = self._get_sequence_list()

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
        return Sequence(sequence_name, frames_list, 'gmtd_sot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list


class GMTDMotDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gmtd_dir
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)): # ashbin1-1
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  set(clean_lst)
    
    def get_sequence_list(self):
        multi_view_list = []
        for name in self.clean_list:
            seq_list = [i for i in self.sequence_list if name==i.split('-')[0]]
            multi_view_list.append([self._construct_sequence(s) for s in seq_list])
        return multi_view_list

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

        seq_view_id = int(sequence_name.split('-')[1])
        return Sequence(sequence_name, frames_list, 'gmtd_mot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible,
                        seq_view_id=seq_view_id)

    def __len__(self):
        return len(self.sequence_list)


    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list