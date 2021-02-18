from Models.BaseModel import BaseModule
from Models.Datasets.ImageDataset import ImageDataset

import torch.nn as nn

from os.path import isfile, join
from os import listdir

import random
import torch


class GAN(BaseModule):
    def loss(self, output, target):
        return self.loss_function(output, target)

    def forward(self, x, part):
        if x.is_leaf:
            x.requires_grad = True
        if part == "generator":
            return torch.sigmoid(self.generator(x))
        if part == "discriminator":
            return self.discriminator(x)

    def __init__(self, generator, discriminator, loss_function=nn.L1Loss()):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.loss_function = loss_function

    @staticmethod
    def load_dataset(image_dir, transforms=None, train_split=0.75, shuffle=True, limit=None):
        train = ImageDataset(transforms)
        test = ImageDataset(transforms)

        examples = [(image_dir + f, torch.tensor([1.0])) for f in listdir(image_dir) if isfile(join(image_dir, f))]

        for filename, _ in examples:
            if not isfile(filename):
                examples.remove(filename)

        if shuffle:
            random.shuffle(examples)

        count = len(examples)

        if limit is not None:
            count = min(count, limit)

        split = int(count * train_split)

        train.examples = examples[0:split]
        test.examples = examples[split:count]

        return train, test
