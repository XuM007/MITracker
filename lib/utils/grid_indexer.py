import torch

class GridIndexer:
    def __init__(self, space_size, space_center, voxels_per_axis):
        self.space_size = space_size
        self.space_center = space_center
        self.voxels_per_axis = voxels_per_axis
        self.voxel_size = [space_size[i] / voxels_per_axis[i] for i in range(len(space_size))]
        self.grid_start = [space_center[i] - space_size[i] / 2 for i in range(len(space_size))]

    def compute_indices_2d(self, points):
        device = points.device
        indices = torch.floor((points - torch.tensor(self.grid_start[:2], device=device)) / torch.tensor(self.voxel_size[:2], device=device)).int()
        return indices

    def compute_indices_3d(self, points):
        device = points.device
        indices = torch.floor((points - torch.tensor(self.grid_start, device=device)) / torch.tensor(self.voxel_size, device=device)).int()
        return indices