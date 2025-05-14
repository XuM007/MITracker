import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', max_view_num = None,settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.max_view_num = max_view_num
        self.mode = mode
        self.settings = settings
        

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box, bbox: tensor([1060.,  857.,  226.,   70.])
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0) 

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], new_roll=False)
                

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            for a in data[s + '_anno']:
                if (a[2:4] < 0).any():
                    data['valid'] = False
                    return data
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']] 

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s]) 
            if (crop_sz < 1).any():
                data['valid'] = False
                return data

            crops, boxes, att_mask, mask_crops, crop_bbox, resize_transform = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s])

            # Apply transforms, resize here
            data[s + '_images'], data[s + '_anno'], data[s + '_att'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, joint=False)
                
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    return data
                
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    return data
            if s == 'search':
                data['search_crop_bbox'] = crop_bbox
                data['search_resize_transform'] = resize_transform
                
        if data['seq_view_id'] is not None:
            views_temp_num = self.settings.num_template * self.max_view_num
            views_sc_num = self.settings.num_search * self.max_view_num
            
            while len(data['template_images']) < views_temp_num:
                data['template_images'].append(torch.zeros_like(data['template_images'][0]))
                data['template_anno'].append(torch.zeros_like(data['template_anno'][0]))
                data['template_att'].append(torch.zeros_like(data['template_att'][0]))
            
            while len(data['search_images']) < views_sc_num:
                data['search_images'].append(torch.zeros_like(data['search_images'][0]))
                data['search_anno'].append(torch.zeros_like(data['search_anno'][0]))
                data['search_att'].append(torch.zeros_like(data['search_att'][0]))
                data['search_crop_bbox'].append(torch.zeros_like(data['search_crop_bbox'][0]))
                data['search_resize_transform'].append(torch.zeros_like(data['search_resize_transform'][0]))
                data['search_visible'].append(torch.zeros_like(data['search_visible'][0]))
            
            while len(data['seq_view_id']) < self.max_view_num:
                data['seq_view_id'].append(0)
            data['seq_view_id'] = torch.tensor(data['seq_view_id'], dtype=torch.long)

        data['valid'] = True
        
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data