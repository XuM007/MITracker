import numpy as np
import torch
import torch.nn.functional as F
import lib.utils.geom as geom
import lib.utils.basic as basic


class VoxelUtil:
    '''From https://github.com/tteepe/TrackTacular/tree/main/WorldTrack/utils'''
    def __init__(self, Y, Z, X, scene_centroid, bounds, pad=None, assert_cube=False):
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        self.Y, self.Z, self.X = Y, Z, X

        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX - self.XMIN) / float(X)
        self.default_vox_size_Y = (self.YMAX - self.YMIN) / float(Y)
        self.default_vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)

        if pad:
            Y_pad, Z_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)) or (
                    not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)):
                print('Y, Z, X', Y, Z, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Z))
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Y))

    def Ref2Mem(self, xyz, Y, Z, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert (C == 3)
        mem_T_ref = self.get_mem_T_ref(B, Y, Z, X, assert_cube=assert_cube, device=device)
        xyz = geom.apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Y, Z, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Y, Z, X, assert_cube=assert_cube, device=xyz_mem.device)
        xyz_ref = geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_mem_T_ref(self, B, Y, Z, X, assert_cube=False, device='cuda'):
        vox_size_X = (self.XMAX - self.XMIN) / float(X) 
        vox_size_Y = (self.YMAX - self.YMIN) / float(Y)
        vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)

        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert (np.isclose(vox_size_X, vox_size_Y))
            assert (np.isclose(vox_size_X, vox_size_Z))

        # translation
        center_T_ref = np.eye(4)
        center_T_ref[0, 3] = -self.XMIN - vox_size_X / 2.0
        center_T_ref[1, 3] = -self.YMIN - vox_size_Y / 2.0
        center_T_ref[2, 3] = -self.ZMIN - vox_size_Z / 2.0
        center_T_ref = torch.tensor(center_T_ref, device=device, dtype=vox_size_X.dtype).view(1, 4, 4).repeat([B, 1, 1])

        # scaling
        mem_T_center = np.eye(4)
        mem_T_center[0, 0] = 1. / vox_size_X
        mem_T_center[1, 1] = 1. / vox_size_Y
        mem_T_center[2, 2] = 1. / vox_size_Z
        mem_T_center = torch.tensor(mem_T_center, device=device, dtype=vox_size_X.dtype).view(1, 4, 4).repeat([B, 1, 1])
        mem_T_ref = basic.matmul2(mem_T_center, center_T_ref)
        return mem_T_ref

    def get_ref_T_mem(self, B, Y, Z, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Y, Z, X, assert_cube=assert_cube, device=device)
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def unproject_image_to_mem(self, rgb_camB, pixB_T_refA, camB_T_refA, Y, Z, X, assert_cube=False, xyz_refA=None,
                               z_sign=1, mode='bilinear'):
        """
        rgb_camB: B*S x 128 x H x W
        pixB_T_refA: B*S x 4 x 4
        camB_T_refA: B*S x 4 x 4
        rgb lives in B pixel coords we want everything in A ref coords
        this puts each C-dim pixel in the rgb_camB along a ray in the voxel grid
        """
        B, C, H, W = rgb_camB.shape
        if xyz_refA is None:
            xyz_memA = basic.gridcloud3d(B, Y, Z, X, norm=False, device=pixB_T_refA.device) # MemA: 0-199, 0-199, 0-2 
            xyz_refA = self.Mem2Ref(xyz_memA, Y, Z, X, assert_cube=assert_cube) # refA:xmin-xmax 1-399; 1-399; 0.5-2.5
        xyz_refA = xyz_refA.double()
        xyz_camB = geom.apply_4x4(camB_T_refA, xyz_refA)
        xyz_pixB = geom.apply_4x4(pixB_T_refA, xyz_refA)
        
        normalizer = torch.unsqueeze(xyz_pixB[:, :, 2], 2)
        EPS = 1e-6
        normalizer[normalizer <= 0] = normalizer[normalizer <= 0].clamp(max=-EPS)
        normalizer[normalizer >= 0] = normalizer[normalizer >= 0].clamp(min=EPS)
        z = xyz_camB[:, :, 2]
        xy_pixB = xyz_pixB[:, :, :2] / normalizer  # B,N,2

        x, y = xy_pixB[:, :, 0], xy_pixB[:, :, 1]  # B,N

        x_valid = (x >= 0).bool() & (x / float(W) <= 1).bool()
        y_valid = (y >= 0).bool() & (y / float(H) <= 1).bool()
        z_valid = (z_sign * z >= 0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Y, Z, X).float()

        # native pytorch version
        y_pixB, x_pixB = basic.normalize_grid2d(y, x, H, W)
        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)  # B*S,128,1(D),H,W
        xyz_pixB = torch.reshape(xyz_pixB, [B, Y, Z, X, 3])  # B*S,200,8,200,3
        values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False, mode=mode)
        values = torch.reshape(values, (B, C, Y, Z, X))
        values = values * valid_mem
        return values