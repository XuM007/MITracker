import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
from lib.test.evaluation import TrackerMulti


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False, save_video_path=None, run_id=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    if tracker_name == 'mitracker':
        tracker = TrackerMulti(tracker_name, tracker_param, "video")
    else:
        tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results, save_video_path= save_video_path, run_id=run_id)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--save_video_path', type=str, default=None, help='Path to save the tracking video.')
    parser.add_argument('--run_id', type=int, default=0, help='Debug level.')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results, args.save_video_path, args.run_id)

if __name__ == '__main__':
    main()