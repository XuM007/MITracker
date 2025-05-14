# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import torch
import numpy as np
import json

def adjust_camera_parameters(camera, crop_info, resize_transform):
    crop_x1, crop_y1, crop_w, crop_h = crop_info.cpu().numpy()
    scale_x, scale_y = resize_transform[0, 0].cpu().numpy(), resize_transform[1, 1].cpu().numpy()
    camera['cx'] = camera['cx'] - crop_x1
    camera['cy'] = camera['cy'] - crop_y1
    
    # 缩放影响焦距和主点
    camera['fx'] = camera['fx'] * scale_x
    camera['fy'] = camera['fy'] * scale_y
    camera['cx'] *= scale_x
    camera['cy'] *= scale_y
    return camera

def _get_cam(camera_files, params_type='synthetic_part'):
    fusecams= {}
    for camera_file in camera_files:
        with open(camera_file) as cfile:
            cameras = json.load(cfile)['cameras']

        for scene in cameras.keys():
            for view in cameras[scene].keys():
                for k, v in cameras[scene][view].items():
                    if params_type == 'synthetic_all':
                        if k in ['R', 'T']:
                            cameras[scene][view][k] = np.array(v)
                        elif k == 'fx': cameras[scene][view][k] = np.array(913)
                        elif k == 'fy': cameras[scene][view][k] = np.array(913)
                        elif k == 'cx': cameras[scene][view][k] = np.array(960)
                        elif k == 'cy': cameras[scene][view][k] = np.array(540)
                        elif k == 'k': cameras[scene][view][k] = np.array([[0], [0], [0], [0], [0], [0]])
                        elif k == 'p': cameras[scene][view][k] = np.array([[0], [0]])
                    elif params_type == 'synthetic_part':
                        if k in ['R', 'T', 'fx', 'fy', 'cx', 'cy']:
                            cameras[scene][view][k] = np.array(v)
                        elif k == 'k': cameras[scene][view][k] = np.array([[0], [0], [0], [0], [0], [0]])
                        elif k == 'p': cameras[scene][view][k] = np.array([[0], [0]])
                    elif params_type == 'real':
                        cameras[scene][view][k] = np.array(v)
                    else:
                        raise ValueError('Unknown camera parameter type: %s' % params_type)
        fusecams.update(cameras)
    return fusecams

def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3,k4,k5,k6].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    x = R @ X + t
    invalid_z = (x[2, :] == 0)
    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0,:]**2 + x[1,:]**2
    ttt = (1 + Kd[0]*r + Kd[1]*r**2 + Kd[4]*r**3)/(1 + Kd[5]*r + Kd[6]*r**2 + Kd[7]*r**3)
    
    x[0,:] = x[0,:] * ttt + 2 * Kd[2] * x[0,:] * x[1,:] + Kd[3] * (r + 2 * x[0,:]**2)
    x[1,:] = x[1,:] * ttt + 2 * Kd[3] * x[0,:] * x[1,:] + Kd[2] * (r + 2 * x[1,:]**2)
    x[0,:] = K[0,0] * x[0,:] + K[0,1] * x[1,:] + K[0,2]
    x[1,:] = K[1,0] * x[0,:] + K[1,1] * x[1,:] + K[1,2]
    

    x[0:2, x[2,:] < 0] = -1921
    x[0:2, invalid_z] = -1921
    return x

def project_pose(x, camera):
    device = x.device
    
    fx, fy, cx, cy = camera['fx'], camera['fy'], camera['cx'], camera['cy']
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=device)
    t = torch.as_tensor(camera['T'], dtype=torch.float, device=device).view(-1, 1)
    camera_matrix = torch.tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), dtype=torch.float, device=device)
    dist_coeffs = torch.tensor(np.array([camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], \
                                         camera['k'][2], camera['k'][3], camera['k'][4], camera['k'][5]]), dtype=torch.float, device=device)

    projected_points = projectPoints(x.T, camera_matrix, R, t, dist_coeffs)
    projected_points = projected_points.squeeze()
    
    return projected_points[:2].T