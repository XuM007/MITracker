import numpy as np
import torch

def gaussian2Dbev(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = torch.tensor(gaussian2Dbev((diameter, diameter), sigma=sigma))

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def get_bev_gt(mem_pts, Y, X):
    B, _ = mem_pts.shape
    center = torch.zeros((B, Y, X), dtype=torch.float32)

    for pts, c in zip(mem_pts, center):
        ct = pts[:2]
        ct_int = ct.int()

        if ct_int[0] < 0 or ct_int[0] >= X or ct_int[1] < 0 or ct_int[1] >= Y:
            continue

        draw_umich_gaussian(c, ct_int, 1.5)

    return center

def generate_cube_map(shape, centers, sigma=3.0):
    '''
    Generate a batch of cube maps with Gaussian distribution centered at provided centers for multiple sets.
    
    Parameters:
    - shape: (list or tuple) dimensions of the cube map, e.g., [80, 80, 30]
    - centers: (PyTorch tensor) batch of centers for multiple sets, dimensions [N, B, 3]
    - sigma: (float) controls the spread of the Gaussian distribution
    
    Returns:
    - (PyTorch tensor) batch of cube maps, dimensions [N, B, 80, 80, 30]
    '''
    # Determine the device from centers tensor
    device = centers.device

    # Create a meshgrid for the dimensions using PyTorch and move it to the correct device
    x, y, z = torch.meshgrid(
        torch.linspace(0, shape[0]-1, shape[0], device=device),
        torch.linspace(0, shape[1]-1, shape[1], device=device),
        torch.linspace(0, shape[2]-1, shape[2], device=device),
        indexing='ij'
    )

    # Create an empty tensor to store the Gaussian maps on the same device
    gauss_maps = torch.zeros((centers.shape[0], centers.shape[1], *shape), device=device)

    # Generate a Gaussian map for each set of centers
    for n in range(centers.shape[0]):  # Loop over N sets
        for b in range(centers.shape[1]):  # Loop over B centers in each set
            x0, y0, z0 = centers[n, b].float()  # Ensure center coords are floats
            gauss_maps[n, b] = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2.0 * sigma ** 2))
    
    return gauss_maps

def generate_bev_heatmap_indexer(centers, heatmap_size):
    """
    Generate heatmaps for given centers with specified Gaussian radius.
    Returns:
        torch.Tensor:list of generated heatmap
    """
    '''
    centers: B x N x 2
    tensor([[[42., 42.]],
        [[42., 40.]]], device='cuda:0')
    '''
    gaussian_maps = []
    for center in centers:
        bs = center.shape[0]
        wh = torch.tensor([[10, 10]]).to(centers.device).repeat(bs, 1)
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, center, 0.7)
        gaussian_maps.append(gt_scoremap.to(center.device))
    return gaussian_maps

def generate_bev_heatmap(centers, heatmap_size, xyrange):
    """
    Generate heatmaps for given centers with specified Gaussian radius.
    Returns:
        torch.Tensor:list of generated heatmap
    """
    centers = (centers + xyrange//2)//(xyrange//heatmap_size)
    '''
    centers: B x N x 2
    tensor([[[42., 42.]],
        [[42., 40.]]], device='cuda:0')
    '''
    gaussian_maps = []
    for center in centers:
        bs = center.shape[0]
        wh = torch.tensor([[10, 10]]).to(centers.device).repeat(bs, 1)
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        center_int = center.round().long()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, center_int, 0.7)
        gaussian_maps.append(gt_scoremap.to(center.device))
    return gaussian_maps


def generate_heatmap(bboxes, patch_size=320, stride=16, min_overlap=0.7):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    heatmap_size = patch_size // stride
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        bbox = single_patch_bboxes * heatmap_size
        wh = bbox[:, 2:]
        centers_int = (bbox[:, :2] + wh / 2).round()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, centers_int, min_overlap)
        gaussian_maps.append(gt_scoremap.to(bbox.device))
    return gaussian_maps

class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        if min_overlap == 0.7:
            radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        else:
            radius = CenterNetHeatMap.get_gaussian_radius_bigger(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius_bigger(box_size, min_overlap):
        width, height = box_size[..., 0], box_size[..., 1]
        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        scale_factor = 0.7  
        return torch.maximum(r1 * scale_factor, torch.tensor(10.0).to(r1.device)) 

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap

def compute_grids(features, strides):
    """
    grids regret to the input image size
    """
    grids = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        shifts_x = torch.arange(
            0, w * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shifts_y = torch.arange(
            0, h * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                          strides[level] // 2
        grids.append(grids_per_level)
    return grids


def get_center3x3(locations, centers, strides, range=3):
    '''
    Inputs:
        locations: M x 2
        centers: N x 2
        strides: M
    '''
    range = (range - 1) / 2
    M, N = locations.shape[0], centers.shape[0]
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)  # M x N x 2
    centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
    strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
    centers_discret = ((centers_expanded / strides_expanded).int() * strides_expanded).float() + \
                      strides_expanded / 2  # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
    return (dist_x <= strides_expanded[:, :, 0] * range) & \
           (dist_y <= strides_expanded[:, :, 0] * range)


def get_pred(score_map_ctr, size_map, offset_map, feat_size):
    max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)

    idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
    size = size_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
    offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

    return size * feat_size, offset
