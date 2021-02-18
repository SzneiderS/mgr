from Training.Trainers import AutoEncoderTrainer
import torch
import torch.nn as nn
import numpy as np

from Models.AutoEncoder import EncoderConv, DecoderConvTranspose, AutoEncoder, DecoderFC, DecoderConvPad, \
    AutoEncoderTwoDecoders, DecoderUpsampleConv, DecoderRecurrentConvTranspose, EncoderFC

import torchvision.transforms as transforms

import datetime
import random
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from Images.HeightMapTransforms import ToNormalHeightmap, ToNormalMap
from Images.Normalization import ImageNorm, ImageToRange
from Images.GaussianBlurTransform import GaussianBlurN2MPixels
from Images.QuantizeTransforms import QuantizeImageN2MColors

if __name__ == "__main__":
    random.seed(2019)
    torch.manual_seed(2019)

    vector_size = 768

    img_edge = 64
    input_img_channels = 1

    decoder_reconstruction_features = np.linspace(10, 100, int((100 - 10)/10) + 1, dtype=np.int)
    encoder_features = np.linspace(100, 10, int((100 - 10)/10) + 1, dtype=np.int)

    img_size = img_edge * img_edge
    enc = EncoderConv([256, 64, 2], input_img_channels, img_edge, vector_size, variational=True)
    # enc2 = EncoderFC([img_size, vector_size * 6, vector_size * 3, vector_size])
    dec1 = DecoderConvTranspose([64, 256], 1, img_edge, vector_size)
    dec2 = DecoderConvPad(decoder_reconstruction_features, 1, img_edge, vector_size)
    dec3 = DecoderFC(
        [vector_size, vector_size * 2, img_size],
        [-1, 1, img_edge, img_edge]
    )
    dec4 = DecoderUpsampleConv([2, 8, 32, 128, 256], 1, img_edge, vector_size, filter_edge=2)
    dec5 = DecoderRecurrentConvTranspose(50, 1, int(img_edge * 0.4), img_edge, vector_size, filter_edge=3)
    net = AutoEncoder(
        enc,
        dec4,
        nn.L1Loss(reduction="sum")
    )

    quantize = QuantizeImageN2MColors(2, 9)
    blur = GaussianBlurN2MPixels(3, 20)

    trans = transforms.Compose([
        # transforms.Resize((img_edge * 2, img_edge * 2), interpolation=Image.NONE),
        transforms.RandomCrop(img_edge),
        transforms.Grayscale(),
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        transforms.ToTensor(),
        ImageToRange(0, 1)
    ])

    limit = None

    train_set, test_set = net.load_dataset("/sdb7/seb/.generated3/", trans, train_split=0.8, shuffle=True, limit=limit)

    trainer = AutoEncoderTrainer(net, train_set, test_set, batch_size=6*6, lr=1e-4)

    trainer.writer = SummaryWriter(
        log_dir='/sdb7/seb/tb_runs/{}/'.format(datetime.datetime.now().strftime("%d%m%Y_%H%M"))
    )

    # trainer.preInputTransform = transforms.Compose([
    #     transforms.RandomChoice([
    #         quantize,
    #         transforms.Compose([quantize, blur]),
    #         transforms.Compose([blur, quantize])
    #     ]),
    #     ImageToRange(-1, 1)
    # ])

    trainer.preInputTransform = ImageToRange(-1, 1)

    # trainer.preInputTransform = transforms.Compose([
    #     # ToNormalHeightmap()
    #     # transforms.RandomApply([GaussianBlurN2MPixels(1, 20)])
    # ])

    trainer.saveName = "../.nets/vae3"
    trainer.loadName = trainer.saveName

    # trainer.preInputTransform = transforms.RandomApply([
    #     transforms.RandomChoice([
    #         quantize,
    #         transforms.Compose([quantize, blur]),
    #         transforms.Compose([blur, quantize])
    #     ])
    # ])

    # import cProfile
    # cProfile.run('trainer.run()', sort="time")

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    trainer.accuracy_threshold = 200

    # trainer.overfit_batch_loss = 16000

    torch.autograd.set_detect_anomaly(True)

    trainer.run()
