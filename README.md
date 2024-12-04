# MITracker: Multi-View Integration for Visual Object Tracking

[Mengjie Xu](https://xum007.github.io/), [Yitao Zhu](https://absterzhu.github.io/), Haotian Jiang, Jiaming Li, [Zhenrong Shen](https://zhenrongshen.github.io/), [Sheng Wang](http://shengwang.link/), Haolin Huang, Xinyu Wang, Han Zhang, Qing Yang, [Qian Wang](https://qianwang.space/)

[Arxiv] [Project Page] [Model] [Raw Results]

https://github.com/user-attachments/assets/37eeaa24-2788-4a89-927c-85184899a7eb


## Getting Started

### Environment Installation
```
conda create -n mittracker python=3.9
conda activate mittracker
bash install.sh
```

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
