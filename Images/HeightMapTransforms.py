from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as fnn
import torch

from . Normalization import normalize_3d_tensor


def get_pixel_bounded(img, coords):
    x, y = coords
    w, h = img.size
    if x < 0:
        x = 0
    if x >= w:
        x = w - 1
    if y < 0:
        y = 0
    if y >= h:
        y = h - 1
    return img.getpixel((x, y))


def normal_map_tensor(tensor):
    tensor = fnn.pad(tensor, [1, 1, 1, 1])
    vals = tensor[0]
    vals_shape = vals.shape
    # corners
    vals[0, 0] = vals[1, 1]
    vals[0, vals_shape[1] - 1] = vals[1, vals_shape[1] - 2]
    vals[vals_shape[0] - 1, 0] = vals[vals_shape[0] - 2, 1]
    vals[vals_shape[0] - 1, vals_shape[1] - 1] = vals[vals_shape[0] - 2, vals_shape[1] - 2]
    # edges
    for i in range(1, vals_shape[0] - 1):
        vals[0, i] = vals[1, i]
        vals[vals_shape[0] - 1, i] = vals[vals_shape[0] - 2, i]
    for i in range(1, vals_shape[1] - 1):
        vals[i, 0] = vals[i, 1]
        vals[i, vals_shape[1] - 1] = vals[i, vals_shape[1] - 2]
    tensor[0] = vals
    kernel = torch.Tensor([
        [[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]],
        [[
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]],
        [[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]]
    ])
    tensor = tensor.view(1, 1, vals_shape[0], vals_shape[1])
    weight = torch.Tensor([0, 0, 1])
    if tensor.is_cuda:
        weight = weight.cuda()
        kernel = kernel.cuda()
    tensor = fnn.conv2d(tensor, kernel, bias=weight)[0]
    tensor = normalize_3d_tensor(tensor)
    return tensor


def normal_map(img):
    if img.mode != "L":
        img = img.convert("L")

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = preprocess(img)
    img_tensor = normal_map_tensor(img_tensor)
    to_img = transforms.ToPILImage()
    img = normalize_3d_tensor(img_tensor)
    return to_img(img)


def to_normal_heightmap(img):
    if img.mode != "L":
        img = img.convert("L")

    normals = normal_map(img)
    normals = normals.convert("RGBA")

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    to_img = transforms.ToPILImage()

    img_tensor = preprocess(img)
    normals_tensor = preprocess(normals)

    normals_tensor[3] = img_tensor[0]
    return to_img(normals_tensor)


def to_heightmap(normalmap):
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    to_img = transforms.ToPILImage()

    normalmap_tensor = preprocess(normalmap)

    img = to_img(normalmap_tensor[3])
    img = img.convert("RGB")
    return img


class ToNormalMap(object):
    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            size = pic.size()
            if len(size) == 4:
                new_pics = torch.zeros(size[0], 3, size[2], size[3])
                if pic.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(pic[i])
                return new_pics
            if len(size) == 3:
                new_pic = torch.zeros(3, size[1], size[2])
                if pic.is_cuda:
                    new_pic = new_pic.cuda()
                new_pic = normal_map_tensor(pic)
                return new_pic
        if isinstance(pic, Image.Image):
            return normal_map(pic)


class ToNormalHeightmap(object):
    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            size = pic.size()
            if len(size) == 4:
                new_pics = torch.zeros(size[0], 4, size[2], size[3])
                if pic.is_cuda:
                    new_pics = new_pics.cuda()
                for i in range(size[0]):
                    new_pics[i] = self(pic[i])
                return new_pics
            if len(size) == 3:
                new_pic = torch.zeros(4, size[1], size[2])
                if pic.is_cuda:
                    new_pic = new_pic.cuda()
                normals = normal_map_tensor(pic)
                new_pic[:3] = normals
                new_pic[3] = pic
                return new_pic
        if isinstance(pic, Image.Image):
            return to_normal_heightmap(pic)


class FromNormalHeightmap(object):
    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, pic):
        to_heightmap(pic)
