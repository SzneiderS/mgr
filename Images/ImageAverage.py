from os.path import isfile, join
from os import listdir

from PIL import Image

import torchvision.transforms as transforms


def image_average_directory(directory: str):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    total = len(files)

    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    result = None
    for file in files:
        with Image.open(file) as img:
            if result is None:
                result = to_tensor(img)
                continue
            else:
                result += to_tensor(img)
    result /= total

    if result is not None:
        return to_image(result)
