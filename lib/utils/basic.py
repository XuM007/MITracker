import numpy as np
import torch
from PIL import Image

EPS = 1e-6


def sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)


def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)


def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert (B == B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B * S] + otherdims)
    return tensor


def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert (BS % B == 0)
    otherdims = shapelist[1:]
    S = int(BS / B)
    tensor = torch.reshape(tensor, [B, S] + otherdims)
    return tensor


def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d - dmin) / (EPS + (dmax - dmin))
    return d


def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a, b) in zip(x.size(), mask.size()):
        # if not b==1:
        assert (a == b)  # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x * mask # Ns*B, Nv, latent_dim, Y,z, X
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean


def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def meshgrid3d(B, Y, Z, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Y x Z x X

    grid_z = torch.linspace(0.0, Z - 1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, 1, Z, 1])
    grid_z = grid_z.repeat(B, Y, 1, X)

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, Y, 1, 1])
    grid_y = grid_y.repeat(B, 1, Z, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Y, Z, 1)

    if norm:
        grid_y, grid_z, grid_x = normalize_grid3d(
            grid_y, grid_z, grid_x, Y, Z, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_y, grid_z, grid_x

def gridcloud3d(B, Y, Z, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_y, grid_z, grid_x = meshgrid3d(B, Y, Z, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0

    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_y, grid_x


def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0 * (grid_z / float(Z - 1)) - 1.0
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0

    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_z, grid_y, grid_x


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = torch.tensor(gaussian2D((diameter, diameter), sigma=sigma))

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
