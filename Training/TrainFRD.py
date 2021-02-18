import random
import torch

from torchvision.transforms import transforms

from Models.FakeRealDiscriminator import FRD

from Images.Normalization import ImageToRange

from Training.Trainers import FRDTrainer

import torch.nn as nn

import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    random.seed(2019)
    torch.manual_seed(2019)

    img_edge = 64

    net = FRD([256, 64, 4], img_edge, 1, nn.L1Loss())

    trans = transforms.Compose([
        # transforms.Resize((img_edge * 2, img_edge * 2), interpolation=Image.NONE),
        transforms.RandomCrop(img_edge),
        transforms.Grayscale(),
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        transforms.ToTensor(),
        ImageToRange(-1, 1)
        # transforms.Lambda(lambda x: x + torch.rand_like(x) * 1.0 - 0.5),
    ])

    limit = 1000

    train_set, test_set = net.load_dataset("/sdb7/seb/.generated3/", trans, train_split=0.8, shuffle=True, limit=limit)

    trainer = FRDTrainer(net, train_set, test_set, batch_size=6*6, lr=1e-3)

    trainer.writer = SummaryWriter(
        log_dir='/sdb7/seb/tb_runs/{}/'.format(datetime.datetime.now().strftime("%d%m%Y_%H%M"))
    )

    # trainer.saveName = "../.nets/frd2"
    trainer.loadName = trainer.saveName

    torch.autograd.set_detect_anomaly(True)

    trainer.accuracy_threshold = 0.01

    trainer.run()
