import time
from datetime import timedelta
import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
from lib.train.data.image_loader import imwrite_indexed
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, restart = False):
    """Saves the output of the tracker."""
    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)

    '''2022.9.18 create new folder for all datasets'''
    if restart:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset+'_restart')):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset+'_restart'))

        base_results_path = os.path.join(tracker.results_dir, seq.dataset+'_restart', seq.name)
        segmentation_path = os.path.join(tracker.segmentation_dir, seq.dataset+'_restart', seq.name)
    else:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))

        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
        segmentation_path = os.path.join(tracker.segmentation_dir, seq.dataset, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')
    
    def save_frame_pos(file, data):
        frame_pos = np.array(data).astype(int)
        np.savetxt(file, frame_pos, delimiter='\t', fmt='%d')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

        elif key == 'init_positions':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_initpos.txt'.format(base_results_path, obj_id)
                    save_frame_pos(timings_file, d)
            else:
                timings_file = '{}_initpos.txt'.format(base_results_path)
                save_frame_pos(timings_file, data)
        
        elif key == 'segmentation':
            assert len(frame_names) == len(data)
            if not os.path.exists(segmentation_path):
                os.makedirs(segmentation_path)
            for frame_name, frame_seg in zip(frame_names, data):
                imwrite_indexed(os.path.join(segmentation_path, '{}.png'.format(frame_name)), frame_seg)



def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8, restart=False):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if isinstance(seq, list):
            for sub_seq in seq:
                if sub_seq.object_ids is None:
                    ''' 2022.9.18 update code '''
                    if restart:
                        bbox_file = '{}/{}_restart/{}.txt'.format(tracker.results_dir, sub_seq.dataset, sub_seq.name)
                    else:
                        bbox_file = '{}/{}/{}.txt'.format(tracker.results_dir, sub_seq.dataset, sub_seq.name)
                    if not os.path.isfile(bbox_file):
                        return False
                else:
                    bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, sub_seq.name, obj_id) for obj_id in sub_seq.object_ids]
                    missing = [not os.path.isfile(f) for f in bbox_files]
                    if sum(missing) > 0:
                        return False
            return True
        else:
            if seq.object_ids is None:
                ''' 2022.9.18 update code '''
                if restart:
                    bbox_file = '{}/{}_restart/{}.txt'.format(tracker.results_dir, seq.dataset, seq.name)
                else:
                    bbox_file = '{}/{}/{}.txt'.format(tracker.results_dir, seq.dataset, seq.name)

                return os.path.isfile(bbox_file)
            else:
                bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
                missing = [not os.path.isfile(f) for f in bbox_files]
                return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return
    if isinstance(seq, list):
        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, [s.name for s in seq]))
    else:
        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    output = tracker.run_sequence(seq, debug=debug, restart=restart)
    sys.stdout.flush()

    if 'time' in output.keys():
        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        if not debug:
            _save_tracker_output(seq, tracker, output, restart=restart)
    else:
        exec_time = sum(output[seq[0].name]['time'])
        num_frames = len(output[seq[0].name]['time'])
        print('FPS: {}'.format(num_frames / exec_time))
        
        if not debug:
            for sub_seq in seq:
                _save_tracker_output(sub_seq, tracker, output[sub_seq.name], restart=restart)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8, restart=False):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
    dataset_start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, restart=restart)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus, restart) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))
