# :star2: [CVPR'2025] - MITracker

<!--[Mengjie Xu](https://xum007.github.io/), [Yitao Zhu](https://absterzhu.github.io/), Haotian Jiang, Jiaming Li, [Zhenrong Shen](https://zhenrongshen.github.io/), [Sheng Wang](http://shengwang.link/), [Haolin Huang](https://scholar.google.com/citations?user=Hr87lqsAAAAJ&hl=zh-CN), Xinyu Wang, Qing Yang, Han Zhang, [Qian Wang](https://qianwang.space/) -->

The official implementation for the CVPR 2025 paper [[MITracker: Multi-View Integration for Visual Object Tracking](https://arxiv.org/pdf/2502.20111)].

[[Project Page](https://mii-laboratory.github.io/MITracker/)] [Model] [Raw Results]

https://github.com/user-attachments/assets/37eeaa24-2788-4a89-927c-85184899a7eb


## Getting Started

<!-- ### Environment Installation
```
conda create -n mitracker python=3.9
conda activate mitracker
bash install.sh
```-->

### Data Preparation
Please download the [MVTrack (coming)], and prepare the data in the following format:
```
data/MVTrack
├── ashbin1/
│   ├── ashbin1-1/
│   │   ├── img
│   │   ├── groundtruth.txt 
│   │   └── invisible.txt
│   ├── ashbin1-2/
│   ├── ...
├── ashbin3
├── bag1
├── basketball5
...
├── calibs.json
├── mvtrack_test_split.txt
├── mvtrack_train_split.txt
└── mvtrack_val_split.txt
```

## Usage
**Coming soon.**

## Acknowledgments
* Thanks to the [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) and [TrackTacular](https://github.com/tteepe/TrackTacular) libraries for enabling quick implementation of our ideas.

## Citation

If you find our paper and code useful for your research and applications, please cite using this BibTeX:
```bibtex 
@article{xu2025mitracker,
  title={MITracker: Multi-View Integration for Visual Object Tracking},
  author={Xu, Mengjie and Zhu, Yitao and Jiang, Haotian and Li, Jiaming and Shen, Zhenrong and Wang, Sheng and Huang, Haolin and Wang, Xinyu and Yang, Qing and Zhang, Han and others},
  journal={arXiv preprint arXiv:2502.20111},
  year={2025}
}
```
