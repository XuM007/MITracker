import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from lib.test.evaluation.multi_object_wrapper import MultiObjectWrapper
from lib.utils.lmdb_utils import decode_img
from lib.utils.box_ops import calc_iou

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [TrackerMulti(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]

class TrackerMulti:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """
    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, result_type=None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.result_type = result_type
        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        return tracker

    def run_sequence(self, seq_list, debug=None, multiobj_mode=None, restart=False):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        if self.run_id is None:
            params = self.get_parameters()
        else:
            params = self.get_parameters(run_id=self.run_id)
        
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_
        params.save_dir = self.results_dir

        # Get init information
        init_info_list = [seq.init_info() for seq in seq_list]
        is_single_object = not seq_list[0].multiobj_mode
         
        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))
        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))
        
        output_dict = self._track_sequence(tracker, seq_list, init_info_list, restart)
        return output_dict

    def _track_sequence(self, tracker, seq_list, init_info_list, restart=False):
        fail_count_dict = {seq.name: 0 for seq in seq_list}
        init_positions_dict = {seq.name: [] for seq in seq_list}

        output_dict = {seq.name: {'target_bbox': [], 'time': [], 'segmentation': []} for seq in seq_list} # all the output results
        def _store_outputs(tracker_out_dict: dict, defaults=None): # update track result to output_dict
            for i,seqname in enumerate(output_dict.keys()):
                default = {} if defaults is None else defaults[seqname]
                for key in output_dict[seqname].keys():
                    val = tracker_out_dict[seqname].get(key, default.get(key, None))
                    if key in tracker_out_dict[seqname] or val is not None:
                        output_dict[seqname][key].append(val)

        # Initialize
        image_list = [self._read_image(seq.frames[0]) for seq in seq_list]
        start_time = time.time()

        tracker.initialize(image_list, init_info_list) 
        out_dict = {seq.name: {} for seq in seq_list}
        prev_output_dict = {seq.name: OrderedDict({}) for seq in seq_list}
        init_default_dict = {seq.name : {'target_bbox': init_info.get('init_bbox'),
                                         'time': (time.time() - start_time)/len(seq_list)} for (seq, init_info) in zip(seq_list, init_info_list)}
        
        
        _store_outputs(out_dict, init_default_dict)
        for frame_num in range(1, len(seq_list[0].frames)):
            frame_paths = [seq.frames[frame_num] for seq in seq_list]
            image_list = [self._read_image(frame_path) for frame_path in frame_paths]
            start_time = time.time()
            info_list = [seq.frame_info(frame_num) for seq in seq_list]

            if restart:
                for info, seq in zip(info_list, seq_list):
                    if fail_count_dict[seq.name] > 10 and seq.target_visible[frame_num]==1:
                        init_positions_dict[seq.name].append(frame_num)
                        info['previous_output'] = {'target_bbox':seq.ground_truth_rect[frame_num]}
                        fail_count_dict[seq.name] = 0
                    else:
                        info['previous_output'] = prev_output_dict[seq.name]
            else:
                for info, seq in zip(info_list, seq_list):
                    info['previous_output'] = prev_output_dict[seq.name] 
                
            out_list = tracker.track(image_list, info_list)
            out_dict = {seq.name: out for seq, out in zip(seq_list, out_list)}
            prev_output_dict = {seq.name: OrderedDict(out) for seq, out in zip(seq_list, out_list)}
            time_dict = {seq.name: {'time': (time.time() - start_time)/len(seq_list)} for seq in seq_list}
            _store_outputs(out_dict, time_dict)

            if restart:
                for seq in seq_list:
                    if seq.target_visible[frame_num]==1:
                        iou = calc_iou(out_dict[seq.name]['target_bbox'], seq.ground_truth_rect[frame_num])
                        if iou < 0.5:
                            fail_count_dict[seq.name] += 1
                        else:
                            fail_count_dict[seq.name] = 0

        for key in ['target_bbox', 'all_boxes', 'all_scores', 'segmentation']:
            for output in output_dict.values():
                if key in output and len(output[key]) <= 1:
                    output.pop(key)

        if restart:
            for seq in seq_list:
                output_dict[seq.name]['init_positions'] = init_positions_dict[seq.name]

        return output_dict

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False, save_video_path=None, run_id = None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters(run_id=run_id)

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        frame_rate = cap.get(cv.CAP_PROP_FPS)
        
        success, frame = cap.read()
        
        def _build_init_info(box):
            return {'init_bbox': box, 'seq_view_id': 0}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize([frame], [_build_init_info(optional_box)])
            output_boxes.append(optional_box)
        else:
            raise ValueError("optional_box must be provided when running in non-interactive mode")

        if save_video_path is not None:
            save_video_dir = os.path.dirname(save_video_path)
            if not os.path.exists(save_video_dir):
                os.makedirs(save_video_dir)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            outvideo = cv.VideoWriter(save_video_path, fourcc, frame_rate, (frame.shape[1], frame.shape[0]))
            first_frame = frame.copy()
            cv.rectangle(first_frame, (int(optional_box[0]), int(optional_box[1])), (int(optional_box[2] + optional_box[0]), int(optional_box[3] + optional_box[1])),
                         (0, 255, 0), 2)
            outvideo.write(first_frame)
            
        info_list = [{}]
        prev_output_dict = {'0': OrderedDict({})}
        
        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            info_list[0]['previous_output'] = prev_output_dict['0']
            out_list = tracker.track([frame], info_list)
            prev_output_dict = {'0': OrderedDict(out_list[0])}
            state = [int(s) for s in out_list[0]['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 2)

            if save_video_path is not None:
                out_frame = frame_disp.copy()
                outvideo.write(out_frame)
            
        # When everything done, release the capture
        cap.release()
       
        if save_video_path is not None:
            outvideo.release()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
            
    def get_parameters(self, run_id=None):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        if run_id is None:
            params = param_module.parameters(self.parameter_name)
        else:
            params = param_module.parameters(self.parameter_name, run_id)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")