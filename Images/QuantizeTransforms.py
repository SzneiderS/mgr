import torch
from PIL import Image
from random import randint

from Images.shared import tensorize, to_img
from Images.Normalization import image_to_range


class QuantizeImageN2MColors(object):
    def __init__(self, min_colors: int, max_colors: int = None):
        assert min_colors >= 2
        self.min_colors = min_colors
        self.max_colors = self.min_colors if max_colors is None else max_colors
        assert self.max_colors >= min_colors
        assert self.max_colors <= 255

    def __repr__(self):
        return self.__class__.__name__ + '({0}, max_colors={1})'.format(self.min_colors, self.max_colors)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = self(tensorize(img))
            return to_img(img)
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 3:
                min_img = img.min()
                max_img = img.max()
                img = image_to_range(img, 0, 1)
                n = randint(self.min_colors, self.max_colors) - 1
                img *= (n + 1) * 0.999
                original_dtype = img.dtype
                img = img.to(dtype=torch.uint8)
                img = img.to(dtype=original_dtype)
                img /= n
                img = image_to_range(img, min_img, max_img)
                return img
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(img[i])
                return new_pics
