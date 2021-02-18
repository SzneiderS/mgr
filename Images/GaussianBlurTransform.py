import torch
from math import pi, sqrt, exp
from PIL import Image
from Images.shared import tensorize, to_img
import torch.nn.functional as fnn
from itertools import product
from random import randint


def gauss_2d_function(sigma: float, x: float, y: float):
    sigma_part = 2.0 * (sigma ** 2)
    return (1.0 / sqrt(sigma_part * pi)) * exp(-((x ** 2 + y ** 2) / sigma_part))


def gaussian_blur_tensor_padded(tensor: torch.Tensor, sigma: float):
    pixels_from_sigma = int(sigma * 3.0) * 2 + 1
    pixels_half = int(pixels_from_sigma * 0.5)

    blur_tensor = torch.ones(pixels_from_sigma, pixels_from_sigma)
    for x, y in product(range(pixels_from_sigma), range(pixels_from_sigma)):
        blur_tensor[y][x] = gauss_2d_function(sigma, x - pixels_half, y - pixels_half)
    blur_tensor /= blur_tensor.sum()

    size = tensor.size()
    kernel = torch.zeros(size[0], size[0], pixels_from_sigma, pixels_from_sigma)
    for c in range(size[0]):
        kernel[c] = blur_tensor / size[0]

    if tensor.is_cuda:
        kernel = kernel.cuda()

    tensor = tensor.view(1, size[0], size[1], size[2])
    tensor = fnn.conv2d(tensor, kernel)[0]
    return tensor


class GaussianBlurPixels(object):
    def __init__(self, pixels):
        self.pixels = pixels
        self.sigma = self.pixels / 3.0

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.pixels)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = tensorize(img)
            img = self(img)
            return to_img(img)
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 3:
                padding = torch.nn.ReplicationPad2d(self.pixels)
                img = img.view(1, size[0], size[1], size[2])
                img = padding(img)
                img = img.view(size[0], size[1] + 2 * self.pixels, size[2] + 2 * self.pixels)
                img = gaussian_blur_tensor_padded(img, self.sigma)
                # img = img[:, self.pixels:-self.pixels, self.pixels:-self.pixels]
                return img
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(img[i])
                return new_pics


class GaussianBlurN2MPixels(object):
    def __init__(self, min_pixels, max_pixels):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __repr__(self):
        return self.__class__.__name__ + '({0}, {1})'.format(self.min_pixels, self.max_pixels)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = tensorize(img)
            img = self(img)
            return to_img(img)
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 3:
                pixels = randint(self.min_pixels, self.max_pixels)
                sigma = pixels / 3.0
                padding = torch.nn.ReplicationPad2d(pixels)
                img = img.view(1, size[0], size[1], size[2])
                img = padding(img)
                img = img.view(size[0], size[1] + 2 * pixels, size[2] + 2 * pixels)
                img = gaussian_blur_tensor_padded(img, sigma)
                # img = img[:, self.pixels:-self.pixels, self.pixels:-self.pixels]
                return img
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(img[i])
                return new_pics

