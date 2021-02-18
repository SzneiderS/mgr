import torch
from PIL import Image
import torchvision.transforms.functional as fvision
from Images.shared import tensorize, to_img


class NormalizeImageBatch(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 3:
                return fvision.normalize(img, self.mean, self.std, self.inplace)
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = fvision.normalize(img[i], self.mean, self.std, self.inplace)
                return new_pics


def image_to_range(img, min_val=0, max_val=1):
    if isinstance(img, Image.Image):
        img = image_to_range(tensorize(img), min_val, max_val)
        return to_img(img)
    if isinstance(img, torch.Tensor):
        min_img = img.clone().min()
        img -= min_img
        max_img = img.clone().max()
        if max_img != 0.0:
            img /= max_img

        value_range = max_val - min_val

        img *= value_range
        img += min_val

        return img


class ImageToRange(object):
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = self(tensorize(img))
            return to_img(img)
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(img[i])
                return new_pics
            if len(size) == 3:
                return image_to_range(img, self.min_val, self.max_val)


class ImageNorm(object):
    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 3:
                mean = img.mean()
                std = img.std()
                img -= mean
                img /= std
                return img
            if len(size) == 4:
                new_pics = torch.zeros(size)
                if img.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(img[i])
                return new_pics


def normalize_color(rgb):
    r, g, b = rgb
    dist = (r ** 2 + g ** 2 + b ** 2) ** 0.5
    if dist != 0.0:
        r /= dist
        g /= dist
        b /= dist
    return r, g, b


def normalize_3d_tensor(tensor):
    width, height = tensor.shape[1:]
    tensor = tensor.view(3, -1)
    r2 = tensor[0] ** 2
    g2 = tensor[1] ** 2
    b2 = tensor[2] ** 2
    sum_sqr = (r2 + g2 + b2) ** 0.5
    tensor /= sum_sqr
    tensor = tensor.view(3, height, width)
    return tensor * 0.5 + 0.5


def normalize_tensor(tensor: torch.Tensor, minimum=None, maximum=None):
    if minimum is None:
        minimum = torch.min(tensor)
    if maximum is None:
        maximum = torch.max(tensor)
    tensor_out = tensor.clone()
    tensor_out -= minimum
    value_range = maximum - minimum
    if value_range != 0 and value_range != 1.0:
        tensor_out /= value_range
    return tensor_out


def normalize_img(img: Image, minimum=None, maximum=None):
    if not(img.mode == "L" or img.mode == "F"):
        img = img.convert("L")
    return to_img(normalize_tensor(tensorize(img), minimum, maximum))


def normalize(img, minimum=None, maximum=None):
    if isinstance(img, Image.Image):
        return normalize_img(img, minimum, maximum)
    if isinstance(img, torch.Tensor):
        return normalize_tensor(img, minimum, maximum)