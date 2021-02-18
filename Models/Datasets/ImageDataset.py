from torch.utils.data.dataset import Dataset
from PIL import Image


class ImageDataset(Dataset):
    """
    Simple image dataset. Helpful for models that use image as an input and some arbitrary value as an output.
    The output can be image as well.

    The examples are defined as a tuple (input, output).
    Inputs are defined as paths to images.
    Outputs can be anything.
    """
    def __init__(self, transforms=None):
        """
        transforms - transforms applied to input
        :param transforms:
        """
        self.examples = []
        self.transforms = transforms

    def __getitem__(self, item):
        img, target = self.examples[item]
        img = Image.open(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.examples)
