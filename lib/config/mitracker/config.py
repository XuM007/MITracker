from easydict import EasyDict as edict
import yaml
from lib.train.admin import env_settings
"""
Add default config for MITracker.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "dinov2_vitb14_pretrain.pth"
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "dinov2_vitb14_pretrain"
cfg.MODEL.BACKBONE.STRIDE = 14
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.TOKEN_LEN = 1
cfg.MODEL.BACKBONE.ATTN_TYPE = 'concat'
cfg.MODEL.BACKBONE.HEATMAP_CHANNEL = 32

cfg.MODEL.PRETRAIN_PTH = ''
cfg.MODEL.PRETRAIN_LAYER = ['backbone']

cfg.MODEL.PROJECT = edict()
cfg.MODEL.PROJECT.RESOLUTION = [200, 3, 200] 
cfg.MODEL.PROJECT.BOUNDS = [0, 400, 0, 400, 0, 3]  
cfg.MODEL.PROJECT.SCENE_CENTROID = [0.0, 0.0, 0.0]
cfg.MODEL.PROJECT.WORLD_SHAPE = [8, 8, 3] 
cfg.MODEL.PROJECT.CAMERAS = 'real'
cfg.MODEL.PROJECT.CAMERA_PATH_MVTRACK = f'{env_settings().mvtrack_dir}/calibs.json'
cfg.MODEL.PROJECT.ZSIGN = 1

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0

cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.BBOX_TASK = False

cfg.TRAIN.BEV_WEIGHT = 0.1
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.SAVE_EVERY_EPOCH = False
cfg.TRAIN.SAVE_START_EPOCH = 0

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["MVTrack_train", "GOT10K_train_full"]
cfg.DATA.TRAIN.DATASETS_SAMPLER = 'sot'
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
cfg.DATA.TRAIN.DATASETS_MAXVIEW = 4
cfg.DATA.TRAIN.SAMPLE_MINVIEW = 2

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["MVTrack_val"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.TEMPLATE_NUMBER = 1
cfg.TEST.MEMORY_THRESHOLD = 1000
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.CAMERA_PATH_MVTRACK = f'{env_settings().mvtrack_dir}/calibs.json'
cfg.TEST.CAMERAS = 'real'
cfg.TEST.RESOLUTION = [200, 3, 200] 
cfg.TEST.BOUNDS = [0, 400, 0, 400, 0, 3]  # xmin,xmax,ymin,ymax,zmin,zmax
cfg.TEST.SCENE_CENTROID = [0.0, 0.0, 0.0]
cfg.TEST.WORLD_SHAPE = [8, 8, 3] 

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)