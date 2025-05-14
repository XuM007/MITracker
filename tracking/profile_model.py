import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='odtrack', 
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_384_ep300', help='yaml configure file name')
    args = parser.parse_args()

    return args

def evaluate_mitracker(model, template, search, search_resize_transform, search_crop_bbox, scene_names, seq_view_id):
    '''Speed Test'''
    # Print memory allocated and reserved on the GPU if available
    
    macs1, params1 = profile(model, inputs=(template, search, search_resize_transform, search_crop_bbox, seq_view_id, scene_names),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search, search_resize_transform, search_crop_bbox, seq_view_id, scene_names)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search, search_resize_transform, search_crop_bbox, seq_view_id, scene_names)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
      
def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "mitracker":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_mitracker
        model = model_constructor(cfg, training=False)
        
        template = torch.randn(1, 4, bs, 3, z_sz, z_sz)
        search = torch.randn(1, 4, bs, 3, x_sz, x_sz)
        search_resize_transform = torch.randn(1, 4, bs, 2, 3)
        search_crop_bbox = torch.randn(1, 4, bs, 4)
        scene_names = ['scene1v1']
        seq_view_id = torch.tensor([0, 1, 2, 3]).unsqueeze(0)
        
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        search_resize_transform = search_resize_transform.to(device)
        search_crop_bbox = search_crop_bbox.to(device)
        seq_view_id = seq_view_id.to(device)
        evaluate_mitracker(model, template, search, search_resize_transform, search_crop_bbox, scene_names, seq_view_id)
    else:
        raise NotImplementedError
