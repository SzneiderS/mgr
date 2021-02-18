from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from os.path import isfile, join
from os import listdir

import random
import numpy

from typing import List

from Models.Datasets.ImageDataset import ImageDataset
from . BaseModel import BaseModule

from . import UF

from torch.utils.checkpoint import checkpoint


class AutoEncoderDataset(ImageDataset):
    def __getitem__(self, item):
        filename = self.examples[item]
        img = Image.open(filename)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img


class Decoder(BaseModule, ABC):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode_module = None


class DecoderFC(Decoder):
    def __init__(self, units, shape):
        if isinstance(units, numpy.ndarray) or isinstance(units, torch.Tensor):
            units = units.tolist()
        assert len(units) > 0, "Units list should have at least one element"
        super(DecoderFC, self).__init__()

        decode_layers = []

        for i in range(1, len(units)):
            prev_layer = int(units[i - 1])
            curr_layer = int(units[i])

            fc = nn.Linear(prev_layer, curr_layer)
            torch.nn.init.xavier_normal_(fc.weight)
            if i < len(units) - 1:
                decode_layers.append(nn.Sequential(
                    fc,
                    nn.LeakyReLU(0.1),
                    nn.BatchNorm1d(curr_layer)
                ))
            else:
                decode_layers.append(fc)

        self.decode_module = nn.ModuleList(decode_layers)
        self.vector_size = units[0]

        if isinstance(shape, list):
            shape = torch.Size(shape)
        self.shape = shape

    def forward(self, x):
        assert self.decode_module is not None, "Decoder should have a decode module"
        for _, l in enumerate(self.decode_module):
            x = l(x)
        x = x.view(self.shape)
        return x


class DecoderConvTranspose(Decoder):
    def __init__(self, features: List[int], channels, img_edge, vector_size):
        if isinstance(features, numpy.ndarray) or isinstance(features, torch.Tensor):
            features = features.tolist()
        assert len(features) > 0, "Features list should have at least one element"
        super(DecoderConvTranspose, self).__init__()

        decode_layers = []

        features.append(channels)
        self.img_edge = img_edge

        for i in range(1, len(features)):
            prev_features = int(features[i - 1])
            curr_features = int(features[i])

            conv_transpose = nn.ConvTranspose2d(prev_features, curr_features, 2, stride=2)
            if i < len(features) - 1:
                decode_layers.append(nn.Sequential(
                    conv_transpose,
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.BatchNorm2d(curr_features, 0.1)
                ))
            else:
                decode_layers.append(conv_transpose)

        self.vector_size = vector_size

        self.decode_module = nn.ModuleList(decode_layers)

        self.first_feature = int(features[0])

        self.img_edge_fc = self.img_edge
        for _ in range(len(features) - 1):
            self.img_edge_fc //= 2

        self.img_to_fc = self.img_edge_fc ** 2 * self.first_feature

        self.fc = nn.Linear(vector_size, self.img_to_fc)

    def forward(self, x):
        assert self.decode_module is not None, "Decoder should have a decode module"
        x = self.fc(x)
        x = x.view(-1, self.first_feature, self.img_edge_fc, self.img_edge_fc)
        for i in range(0, len(self.decode_module)):
            x = self.decode_module[i](x)
        x = F.interpolate(x, (self.img_edge, self.img_edge), mode="bilinear", align_corners=True)
        return x


class DecoderRecurrentConvTranspose(Decoder):
    def __init__(self, feature, img_channels, start_img_edge, final_img_edge, vector_size, filter_edge=3):
        assert feature > 0, "Feature should be at least 1"
        assert  filter_edge >= 2, "Filter edge should be greater or equal to 3"
        super(DecoderRecurrentConvTranspose, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(feature, feature, filter_edge),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(feature),
        )
        self.conv_final = nn.Conv2d(feature, img_channels, 1)

        self.feature = feature

        self.start_img_edge = start_img_edge
        self.final_img_edge = final_img_edge

        img_to_fc = start_img_edge ** 2 * feature
        self.fc = nn.Linear(vector_size, img_to_fc)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.feature, self.start_img_edge, self.start_img_edge)
        size = x.size()
        while size[2] < self.final_img_edge:
            x = self.conv_transpose(x)
            size = x.size()
        x = self.conv_final(x)
        x = F.interpolate(x, (self.final_img_edge, self.final_img_edge), mode="bilinear", align_corners=True)
        return x


class DecoderConvPad(Decoder):
    def __init__(self, features, channels, img_edge, vector_size):
        if isinstance(features, numpy.ndarray) or isinstance(features, torch.Tensor):
            features = features.tolist()
        assert len(features) > 0, "Features list should have at least one element"
        super(DecoderConvPad, self).__init__()

        decode_layers = []

        features.append(channels)
        self.img_edge = img_edge

        for i in range(1, len(features)):
            prev_features = int(features[i - 1])
            curr_features = int(features[i])

            conv_padding = nn.Conv2d(prev_features, curr_features, 3, stride=1, padding=1)
            if i < len(features) - 1:
                decode_layers.append(nn.Sequential(conv_padding, nn.PReLU()))
            else:
                decode_layers.append(conv_padding)

        self.decode_module = nn.ModuleList(decode_layers)

        self.first_feature = int(features[0])

        # self.img_edge += (len(features) - 1) * 4

        self.img_to_fc = self.img_edge ** 2 * self.first_feature

        self.fc = nn.Linear(vector_size, self.img_to_fc)

    def forward(self, x):
        assert self.decode_module is not None, "Decoder should have a decode module"
        x = self.fc(x)
        x = x.view(-1, self.first_feature, self.img_edge, self.img_edge)
        for l in self.decode_module:
            x = l(x)
        return x


class DecoderUpsampleConv(Decoder):
    def forward(self, x):
        assert self.decode_module is not None, "Decoder should have a decode module"
        x = self.fc(x)
        x = x.view(-1, self.first_feature, self.img_edge, self.img_edge)
        for l in self.decode_module:
            x = l(x)
        return x

    def __init__(self, features: List[int], channels, img_edge, vector_size, filter_edge=3):
        if isinstance(features, numpy.ndarray) or isinstance(features, torch.Tensor):
            features = features.tolist()
        assert len(features) > 0, "Features list should have at least one element"
        assert filter_edge > 0, "Filter should be at least 1x1"
        assert isinstance(filter_edge, int), "Filter edge should be int"
        super(DecoderUpsampleConv, self).__init__()

        decode_layers = []

        # features.append(channels)
        self.img_edge = img_edge

        for i in range(1, len(features)):
            prev_features = int(features[i - 1])
            curr_features = int(features[i])

            upsampling = nn.UpsamplingBilinear2d((self.img_edge + filter_edge - 1, self.img_edge + filter_edge - 1))
            conv = nn.Conv2d(prev_features, curr_features, filter_edge)
            torch.nn.init.xavier_normal_(conv.weight)
            if i < len(features) - 1:
                decode_layers.append(nn.Sequential(
                    upsampling,
                    conv,
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.BatchNorm2d(curr_features)
                ))
            else:
                decode_layers.append(nn.Sequential(upsampling, conv))

        last_feature = int(features[-1])
        conv = nn.Conv2d(last_feature, channels, 1)

        decode_layers.append(conv)

        self.decode_module = nn.ModuleList(decode_layers)

        self.first_feature = int(features[0])

        self.img_edge_fc = self.img_edge

        self.img_to_fc = self.img_edge_fc ** 2 * self.first_feature

        self.vector_size = vector_size
        self.fc = nn.Linear(vector_size, self.img_to_fc)
        torch.nn.init.xavier_normal_(self.fc.weight)


class Encoder(BaseModule, ABC):
    def __init__(self, variational=False):
        super(Encoder, self).__init__()
        self.encode_module = nn.Module()
        self.fc1 = nn.Module()
        self.fc2 = nn.Module()
        self.variational = variational


class EncoderFC(Encoder):
    def __init__(self, units: List[int], shape=None, variational=False):
        if isinstance(units, numpy.ndarray):
            units = units.tolist()
        assert len(units) > 0, "Units list should have at least one element"
        super(EncoderFC, self).__init__(variational)

        encode_layers = []

        self.variational = variational

        for i in range(1, len(units) - 1):
            prev_layer = int(units[i - 1])
            curr_layer = int(units[i])

            fc = nn.Linear(prev_layer, curr_layer)
            torch.nn.init.xavier_normal_(fc.weight)
            encode_layers.append(nn.Sequential(
                fc,
                nn.LeakyReLU(0.1),
                nn.InstanceNorm1d(curr_layer)
            ))

        self.encode_module = nn.ModuleList(encode_layers)

        self.fc1 = nn.Sequential(
            nn.Linear(units[-2], units[-1]),
            nn.BatchNorm1d(units[-1])
        )
        if self.variational:
            self.fc2 = nn.Sequential(
                nn.Linear(units[-2], units[-1]),
                nn.BatchNorm1d(units[-1])
            )

        if shape is None:
            self.shape = torch.Size([-1, units[0]])
        else:
            if isinstance(shape, list):
                shape = torch.Size(shape)
            self.shape = shape

    def forward(self, x):
        assert self.encode_module is not None, "Encoder should have a encode module"
        x = x.view(self.shape)
        for l in self.encode_module:
            x = l(x)
        if self.variational:
            return self.fc1(x), self.fc2(x)
        else:
            return self.fc1(x)


class EncoderConv(Encoder):
    def __init__(self, features: List[int], channels, img_edge, vector_size, variational=False):
        if isinstance(features, numpy.ndarray) or isinstance(features, torch.Tensor):
            features = features.tolist()
        assert len(features) > 0, "Features list should have at least one element"
        super(EncoderConv, self).__init__(variational)

        self.img_edge = img_edge
        self.img_channels = channels

        self.features = features

        features.insert(0, self.img_channels)

        encode_layers = []

        for i in range(1, len(features)):
            prev_features = features[i - 1]
            curr_features = features[i]

            conv = nn.Conv2d(prev_features, curr_features, 3)
            torch.nn.init.xavier_normal_(conv.weight)

            encode_layers.append(nn.Sequential(
                conv,
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(curr_features)
            ))

        self.encode_module = nn.ModuleList(encode_layers)

        self.last_conv_features = features[-1]

        self.img_edge_convolved = img_edge
        for i in range(1, len(features)):
            self.img_edge_convolved = int(UF.calculate_output(self.img_edge_convolved, 3) / 1)
        self.img_to_fc = self.img_edge_convolved ** 2 * self.last_conv_features

        self.fc1 = nn.Linear(self.img_to_fc, vector_size)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        if self.variational:
            self.fc2 = nn.Linear(self.img_to_fc, vector_size)
            torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        assert self.encode_module is not None, "Encoder should have a encode module"
        for l in self.encode_module:
            x = l(x)
        x = x.view(-1, self.img_to_fc)
        if self.variational:
            return self.fc1(x), self.fc2(x)
        else:
            return self.fc1(x)
   
        
class AutoEncoder(BaseModule):
    def __init__(self, enc, dec, loss_function=nn.L1Loss()):
        super(AutoEncoder, self).__init__()
        self.encoder = enc
        self.decoder = dec

        self.loss_function = loss_function

    @staticmethod
    def load_dataset(real_image_dir, transforms=None, train_split=0.75, shuffle=False, limit=None):
        train = AutoEncoderDataset(transforms)
        test = AutoEncoderDataset(transforms)

        examples = [real_image_dir + f for f in listdir(real_image_dir) if isfile(join(real_image_dir, f))]

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    @staticmethod
    def reparametrize(mu, logvar, training=False):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x.requires_grad = True
        cp = checkpoint(lambda v: self.encode(v), x)
        if self.encoder.variational:
            mu, logvar = cp
            cp = self.reparametrize(mu, logvar, self.training)
            return torch.sigmoid(self.decode(cp)), (mu, logvar)
        else:
            return torch.sigmoid(self.decode(torch.tanh(cp))), cp

    def loss(self, output, target):
        if not self.training:
            return self.loss_function(output, target)

        vec = None
        if isinstance(output, tuple):
            output, vec = output
        if self.encoder.variational and vec is not None:
            mu, logvar = vec
            img_loss = self.loss_function(output, target)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return img_loss + kld_loss
        else:
            return self.loss_function(output, target)


class AutoEncoderTwoDecoders(AutoEncoder):
    def __init__(self, enc, dec1, dec2, loss_function=nn.L1Loss()):
        super(AutoEncoderTwoDecoders, self).__init__(enc, None, loss_function)
        self.decoder1 = dec1
        self.decoder2 = dec2

    def decode(self, x):
        cp_dec1 = checkpoint(lambda v: self.decoder1(v), x)
        cp_dec2 = checkpoint(lambda v: self.decoder2(v), x)
        return cp_dec1 - cp_dec2

    def load(self, name):
        return not (not self.encoder.load(name + "_encoder") or not self.decoder1.load(
            name + "_decoder1") or not self.decoder2.load(name + "_decoder2"))

    def save(self, name):
        self.encoder.save(name + "_encoder")
        self.decoder1.save(name + "_decoder1")
        self.decoder2.save(name + "_decoder2")

