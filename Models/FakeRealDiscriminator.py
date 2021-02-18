from Models.Datasets.ImageDataset import ImageDataset
from .BaseModel import BaseModule

from PIL import Image
import torch
import torch.nn as nn

import random

from os.path import isfile, join
from os import listdir

import numpy
from Models import UF

from Images.GaussianBlurTransform import GaussianBlurN2MPixels
from Images.QuantizeTransforms import QuantizeImageN2MColors
from torchvision.transforms import transforms

from Images.Normalization import ImageToRange

noiser = transforms.Compose([
    transforms.Lambda(lambda x: x + torch.rand_like(x) * 2.0 - 1.0),
    ImageToRange(-1, 1)
])


class FakeRealDiscriminatorDataset(ImageDataset):
    quantize = QuantizeImageN2MColors(2, 9)
    blur = GaussianBlurN2MPixels(3, 20)

    fake_transform = transforms.RandomChoice([
        quantize,
        transforms.Compose([blur, quantize])
    ])

    def __getitem__(self, item):
        filename = self.examples[item]
        img = Image.open(filename)
        if self.transforms is not None:
            img = self.transforms(img)

        if random.random() > 0.5:
            answer = torch.tensor([1.0])
        else:
            answer = torch.tensor([0.0])
            img = self.fake_transform(img)
            if random.random() > 0.5:
                img = noiser(img)

        return img, answer


class FRD(BaseModule):
    def __init__(self, features, img_edge, img_channels, loss_function=nn.L1Loss()):
        if isinstance(features, numpy.ndarray) or isinstance(features, torch.Tensor):
            features = features.tolist()
        assert len(features) > 0, "Features list should have at least one element"
        super(FRD, self).__init__()

        self.loss_function = loss_function

        self.img_edge = img_edge
        self.img_channels = img_channels

        features.insert(0, self.img_channels)

        conv_layers = []
        for i in range(1, len(features)):
            prev_features = features[i - 1]
            curr_features = features[i]

            conv = nn.Conv2d(prev_features, curr_features, 3)
            torch.nn.init.xavier_normal_(conv.weight)

            conv_layers.append(nn.Sequential(
                conv,
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(curr_features)
            ))

        self.conv_module = nn.ModuleList(conv_layers)

        self.last_conv_features = features[-1]

        self.img_edge_convolved = img_edge
        for i in range(1, len(features)):
            self.img_edge_convolved = int(UF.calculate_output(self.img_edge_convolved, 3) / 2)

        self.img_to_fc = self.img_edge_convolved ** 2 * self.last_conv_features

        fc = nn.Linear(self.img_to_fc, 1)
        torch.nn.init.xavier_normal_(fc.weight)

        self.last_layer = nn.Sequential(
            nn.Dropout(0.25),
            fc
        )

    def loss(self, output, target):
        img_loss = self.loss_function(output, target)
        return img_loss

    @staticmethod
    def load_dataset(image_dir, transforms=None, train_split=0.75, shuffle=True, limit=None):
        train = FakeRealDiscriminatorDataset(transforms)
        test = FakeRealDiscriminatorDataset(transforms)

        examples = [image_dir + f for f in listdir(image_dir) if isfile(join(image_dir, f))]

        for filename in examples:
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

    def forward(self, x):
        for _, l in enumerate(self.conv_module):
            x = l(x)

        x = x.view(-1, self.img_to_fc)

        # x = torch.dropout(x, 0.25, self.training)

        x = self.last_layer(x)

        return torch.sigmoid(x)
