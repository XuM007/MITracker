import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results
from lib.test.evaluation import get_dataset, trackerlist


dataset_name = 'mvtrack_sot_test'

trackers = []

result = 'OPE' # 'R-OPE' or 'OPE'

trackers.extend(trackerlist(name='mitracker_stage1', parameter_name='baseline', dataset_name=dataset_name,run_ids=50, result_type = 'mvtrack_sot', display_name='MITrackerStage1'))
trackers.extend(trackerlist(name='mitracker', parameter_name='baseline', dataset_name=dataset_name,run_ids=40, result_type = 'mvtrack_mot', display_name='MITracker'))

# result = 'R-OPE' # 'R-OPE' or 'OPE'

# trackers.extend(trackerlist(name='mitracker_stage1', parameter_name='baseline', dataset_name=dataset_name,run_ids=50, result_type = 'mvtrack_sot_restart', display_name='MITrackerStage1'))
# trackers.extend(trackerlist(name='mitracker', parameter_name='baseline', dataset_name=dataset_name,run_ids=40, result_type = 'mvtrack_mot_restart', display_name='MITracker'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'), exclude_invalid_frames=False)
plot_results(trackers, dataset, dataset_name, merge_results=False, 
             plot_types=('restart') if result=='R-OPE' else ('success', 'prec', 'norm_prec', 'recovery'),
            skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
