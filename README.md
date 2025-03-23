# :star2: [CVPR'2025] - MITracker

<!--[Mengjie Xu](https://xum007.github.io/), [Yitao Zhu](https://absterzhu.github.io/), Haotian Jiang, Jiaming Li, [Zhenrong Shen](https://zhenrongshen.github.io/), [Sheng Wang](http://shengwang.link/), [Haolin Huang](https://scholar.google.com/citations?user=Hr87lqsAAAAJ&hl=zh-CN), Xinyu Wang, Qing Yang, Han Zhang, [Qian Wang](https://qianwang.space/) -->

The official implementation for the CVPR 2025 paper [[MITracker: Multi-View Integration for Visual Object Tracking](https://arxiv.org/pdf/2502.20111)].

[[Project Page](https://mii-laboratory.github.io/MITracker/)] [Model] [Raw Results]

https://github.com/user-attachments/assets/37eeaa24-2788-4a89-927c-85184899a7eb


## Data Preparation
Please complete [this form](https://docs.google.com/forms/d/e/1FAIpQLSeFml5oIyIKT-R8Biw4aGClNeFhgakRE1dXVIbRIbst7uEMaQ/viewform?usp=header) to request authorization for the non-commercial use of **MVTrack**. Once submitted, you will receive an email containing the download links to access the 80GB dataset. 

Place the tracking datasets and organize the data in the following format:

```
./data
├── MVTrack/
    ├── ashbin1/
    │   ├── ashbin1-1/
    │   │   ├── img
    │   │   ├── attributes.json     # frame level target attributes
    │   │   ├── groundtruth.txt     # x, y, w, h
    │   │   └── invisible.txt       # fully occlusion or out-of-view
    │   ├── ashbin1-2/
    │   ├── ...
    │   └── BEV/
    │       └── xyz_index.txt       # x: [-4000, 4000] (mm), y: [-4000, 4000] (mm), z: [-50, 2950] (mm), voxel indices 
    ├── ashbin3
    ├── bag1
    ├── basketball5
    ...
    ├── calibs.json                 # camera intrinsics and extrinsics (mm)
    ├── test_split.txt
    ├── train_split.txt
    └── val_split.txt
├── got10k/
    ├── test
    ├── train
    └── val
```

Below is an example of the annotations provided by the **MVTrack** dataset.
<p align="center">
  <img width="100%" src="assets/MVTrack.jpg" alt="MVTrack_annotation_sample"/>
</p>

<!--https://github.com/user-attachments/assets/18bba035-9237-482e-bf24-a9f574fffe3f
https://github.com/user-attachments/assets/e1b05a21-d95c-4b36-a953-01adb935cc80 -->

## Usage
**Coming soon.**

## Acknowledgments
* Thanks to the [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) and [TrackTacular](https://github.com/tteepe/TrackTacular) libraries for enabling quick implementation of our ideas.


## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@article{xu2025mitracker,
  title={MITracker: Multi-View Integration for Visual Object Tracking},
  author={Xu, Mengjie and Zhu, Yitao and Jiang, Haotian and Li, Jiaming and Shen, Zhenrong and Wang, Sheng and Huang, Haolin and Wang, Xinyu and Yang, Qing and Zhang, Han and others},
  journal={arXiv preprint arXiv:2502.20111},
  year={2025}
}
```
