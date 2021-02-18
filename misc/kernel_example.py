from PIL import Image
from torchvision.transforms import transforms
from Images.Normalization import ImageToRange
from Images.shared import to_img, tensorize
import torch
import torch.nn.functional as fnn

with Image.open("kernel_example_full_input.png") as img:
    grayscale = transforms.Grayscale()
    to01 = ImageToRange(0, 1)
    img = tensorize(grayscale(img))

    kernel = torch.Tensor([
        [[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]
    ])

    img = img.view(1, 1, img.size(1), img.size(2))
    small_example = torch.Tensor([
        [[
            [0.4, 0.2, 0.6, 0.8, 1.0],
            [0.15, 0.05, 0.3, 0.34, 0.75],
            [0.28, 0.39, 0.65, 0.27, 0.13],
            [0.15, 0.79, 0.5, 0.1, 0.89],
            [0.42, 0.05, 0.66, 0.2, 0.95]
        ]]
    ])

    kernel = kernel.transpose(2, 3)

    if torch.cuda.is_available():
        img = img.cuda()
        kernel = kernel.cuda()
        small_example = small_example.cuda()

    output = fnn.conv2d(img, kernel)[0]
    small_example_output = fnn.conv2d(small_example, kernel)[0]

    if torch.cuda.is_available():
        output = output.cpu()

    output = to_img(to01(output))
    output.save("kernel_example_full_output.png")