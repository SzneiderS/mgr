import torch

from torch.utils.tensorboard import SummaryWriter
from Models.BaseModel import BaseModule
from Models.AutoEncoder import AutoEncoder

import torchvision.transforms as transforms
from Images.Normalization import ImageToRange

import shutil

if __name__ == "__main__":
    log_dir = '/sdb7/seb/tb_runs/ae_embeddings/'
    net_file = '/sdb7/seb/ready/ae1'
    images_dir = '/sdb7/seb/.generated3/'

    shutil.rmtree(log_dir, ignore_errors=True)

    enc = BaseModule.load(net_file).encoder

    writer = SummaryWriter(
        log_dir=log_dir
    )

    img_edge = 64
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
    ])

    to_range = ImageToRange(0, 1)

    limit = 3000

    image_set, _ = AutoEncoder.load_dataset(
        images_dir,
        trans,
        train_split=1.0,
        shuffle=False,
        limit=limit
    )
    image_loader = torch.utils.data.DataLoader(
        image_set,
        batch_size=16,
        drop_last=False,
        shuffle=False,
        num_workers=4
    )

    if torch.cuda.is_available():
        enc = enc.cuda()

    global_vecs = None
    global_inputs = None
    with torch.no_grad():
        for inputs, _ in image_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            vecs = enc.forward(inputs)

            if enc.variational:
                vecs = vecs[0]

            inputs = to_range(inputs)
            if global_inputs is None:
                global_inputs = inputs
            else:
                global_inputs = torch.cat((global_inputs, inputs))

            if global_vecs is None:
                global_vecs = vecs
            else:
                global_vecs = torch.cat((global_vecs, vecs))

        writer.add_embedding(global_vecs, label_img=global_inputs)
