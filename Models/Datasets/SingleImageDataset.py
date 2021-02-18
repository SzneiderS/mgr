from torch.utils.data.dataset import Dataset
from PIL import Image
from os import listdir
from os.path import isfile, join
import random


class SingleImageDataset(Dataset):
    """
    Single image dataset. Helpful for models that use the same image as input and output.

    The examples are defined as paths to images.
    """
    def __init__(self, transforms=None):
        """
        :param transforms: transforms applied to input
        """
        self.examples = []
        self.transforms = transforms

    def __getitem__(self, item):
        img = self.examples[item]
        img = Image.open(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def load_from_dir(image_dir, transforms=None, train_split=0.75, shuffle=False, limit=None):
        """
        Creates train and test dataset from images in a directory
        :param image_dir: directory with images
        :param transforms: transforms applied to images
        :param train_split: how much images from directory should be considered as test set (default = 0.75)
        :param shuffle: whether to shuffle all loaded examples
        :param limit: maximum number of images (before split, defaults to None)
        :return: tuple (train_set, test_set)
        """
        train = SingleImageDataset(transforms)
        test = SingleImageDataset(transforms)

        examples = [image_dir + f for f in listdir(image_dir) if isfile(join(image_dir, f))]

        for example in examples:
            if not isfile(example):
                examples.remove(example)

        if shuffle:
            random.shuffle(examples)

        count = len(examples)

        if limit is not None:
            count = min(count, limit)

        split = int(count * train_split)

        train.examples = examples[0:split]
        test.examples = examples[split:count]

        return train, test
