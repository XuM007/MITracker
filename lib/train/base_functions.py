import torch
from torch.utils.data.distributed import DistributedSampler
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, MVTrack_sot, MVTrack_mot
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR, 
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "MVTrack_train", "MVTrack_val"]
        # Tracking Task
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "MVTrack_train":
            if settings.sampler_type == 'mot':
                datasets.append(MVTrack_mot(settings.env.mvtrack_dir, split='train',image_loader=image_loader, sample_minview=settings.sample_minview))
            elif settings.sampler_type == 'sot':
                datasets.append(MVTrack_sot(settings.env.mvtrack_dir, split='train',image_loader=image_loader))
        if name == "MVTrack_val":
            if settings.sampler_type == 'mot':
                datasets.append(MVTrack_mot(settings.env.mvtrack_dir, split='val',image_loader=image_loader, sample_minview=settings.sample_minview))
            elif settings.sampler_type == 'sot':
                datasets.append(MVTrack_sot(settings.env.mvtrack_dir, split='val',image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    settings.sample_minview = getattr(cfg.DATA.TRAIN, "SAMPLE_MINVIEW", 2)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    settings.sampler_type = getattr(cfg.DATA.TRAIN, "DATASETS_SAMPLER", 1)
    if settings.sampler_type == 'sot':
        dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                                p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                                samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                                max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                                num_template_frames=settings.num_template, processing=data_processing_train,
                                                frame_sample_mode=sampler_mode, train_cls=train_cls)
    elif settings.sampler_type == 'mot':
        settings.max_view_num = cfg.DATA.TRAIN.DATASETS_MAXVIEW
        transform_joint_freeze = tfm.Transform(tfm.ToGrayscale(probability=0.05))
        transform_train_freeze = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
        data_processing_train_freeze = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train_freeze,
                                                       joint_transform=transform_joint_freeze,
                                                       max_view_num=settings.max_view_num,
                                                       settings=settings)
        
        dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                                p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                                samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                                max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                                num_template_frames=settings.num_template, processing=data_processing_train_freeze,
                                                frame_sample_mode=sampler_mode, train_cls=train_cls)
    else:
        raise ValueError("Unsupported sampler type")

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings,
                                                     max_view_num=None if settings.sampler_type == 'sot' else settings.max_view_num)
    
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg, settings):
    # Freeze layers if specified in cfg.MODEL.FREEZE_LAYERS
    if hasattr(cfg.MODEL, "FREEZE_LAYERS") and len(cfg.MODEL.FREEZE_LAYERS) > 0:
        for name, param in net.named_parameters():
            if any(layer in name for layer in cfg.MODEL.FREEZE_LAYERS):
                param.requires_grad = False

    # Visual Encoder
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if not ("backbone" in n or "output_upscaling" in n) and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if ("backbone" in n or "output_upscaling" in n) and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
    ]

    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print("Learnable parameters: ", n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
