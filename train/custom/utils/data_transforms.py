
import random
import torch

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img
2、img的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class to_tensor(object):
    def __call__(self, img):
        img_o = torch.from_numpy(img)
        return img_o

class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img):  
        if self.win_clip is not None:
            img = torch.clip(img, self.win_clip[0], self.win_clip[1])
        img_o = self._norm(img)
        return img_o
    
    def _norm(self, img):
        ori_shape = img.shape
        img_flatten = img.reshape(ori_shape[0], -1)
        img_min = img_flatten.min(dim=-1,keepdim=True)[0]
        img_max = img_flatten.max(dim=-1,keepdim=True)[0]
        img_norm = (img_flatten - img_min)/(img_max - img_min)
        img_norm = (img_norm - 0.5) / 0.5
        img_norm = img_norm.reshape(ori_shape)
        return img_norm


class random_flip(object):
    def __init__(self, axis=1, prob=0.5):
        assert isinstance(axis, int) and axis in [1,2]
        self.axis = axis
        self.prob = prob

    def __call__(self, img):
        img_o = img
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
        return img_o

class random_rotate90(object):
    def __init__(self, k=1, prob=0.5):
        assert isinstance(k, int) and k in [1,2,3]
        self.k = k
        self.prob = prob

    def __call__(self, img):
        img_o = img
        if random.random() < self.prob:
            img_o = torch.rot90(img, self.k, [1, 2])
        return img_o

class random_crop(object):
    def __init__(self, crop_size=[256, 256]):
        self.crop_size = crop_size  # [crop_h, crop_w]

    def __call__(self, img):
        c, h, w = img.shape
        crop_h, crop_w = self.crop_size

        if crop_h > h or crop_w > w:
            raise ValueError(f"Crop size {self.crop_size} should be <= image size {[h, w]}")

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        return img[:, top:top+crop_h, left:left+crop_w]

class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img):
        img_o = img
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if tuple(img.shape[1:]) == tuple(self.size):
            return img
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="bilinear") 
        img_o = img_o.squeeze(0)
        return img_o


class random_center_crop(object):
    def __init__(self, crop_size, shift_range, prob=0.5):
        self.crop_size = crop_size
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, img):
        img_o = img
        d, w = img.shape[1:]
        if d <= self.crop_size[0] or w <= self.crop_size[1]:
            return img_o
        if random.random() < self.prob:
            crop_x_start = min(max(0, (d - self.crop_size[0])//2-random.randint(-self.shift_range[0], self.shift_range[0])), d-self.crop_size[0])
            crop_y_start = min(max(0, (w - self.crop_size[1])//2-random.randint(-self.shift_range[1], self.shift_range[1])), w-self.crop_size[1])
            img_o = img[:, crop_x_start:crop_x_start+self.crop_size[0], crop_y_start:crop_y_start+self.crop_size[1]]     
        return img_o

