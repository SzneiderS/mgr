import random
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from Models.GAN import GAN
from Models.AutoEncoder import DecoderUpsampleConv, DecoderFC, DecoderConvTranspose
from Models.FakeRealDiscriminator import FRD

from Images.Normalization import ImageToRange

from Training.Trainers import GANTrainer

import datetime
from torch.utils.tensorboard import SummaryWriter

import torchvision
from misc.mnist_dataset import MNISTDatasetWrapper

if __name__ == "__main__":
    random.seed(2019)
    torch.manual_seed(2019)

    img_edge = 64
    img_size = img_edge * img_edge

    vector_size = 100

    generator = DecoderUpsampleConv([2, 8, 32, 128, 256], 1, img_edge, vector_size, filter_edge=2)
    generator2 = DecoderFC(
        [vector_size, vector_size * 2, img_size * 2, img_size],
        [-1, 1, img_edge, img_edge]
    )
    generator3 = DecoderConvTranspose([4, 8, 32, 128, 256], 1, img_edge, vector_size)
    discriminator = FRD([64, 256], img_edge, 1)

    net = GAN(generator3, discriminator, nn.BCELoss(reduction="sum"))

    trans = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(img_edge),
        transforms.Grayscale(),
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        transforms.ToTensor(),
        ImageToRange(0, 1)
    ])

    limit = 1000

    train_set, _ = net.load_dataset("/sdb7/seb/.generated3/", trans, train_split=1.0, shuffle=True, limit=limit)
    # train_set = torchvision.datasets.MNIST('../data/', train=True, download=True)
    # train_set = MNISTDatasetWrapper(train_set.data, trans, limit)

    trainer = GANTrainer(net, train_set, None, batch_size=2**6, lr=1e-3)

    trainer.writer = SummaryWriter(
        log_dir='/sdb7/seb/tb_runs/{}/'.format(datetime.datetime.now().strftime("%d%m%Y_%H%M"))
    )

    # trainer.saveName = "../.nets/gan"
    trainer.loadName = trainer.saveName
    # trainer.force_full_load = True

    torch.autograd.set_detect_anomaly(True)

    trainer.run()
