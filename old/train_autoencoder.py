import torch
import torch.optim as optim

import random

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

from Models.AutoEncoder import AutoEncoder, EncoderFC, EncoderConv, DecoderFC, DecoderConvTranspose, DecoderConvPad

from Images.HeightMapTransforms import normalize, ToNormalHeightmap, FromNormalHeightmap, ToNormalMap, NormalizeImageBatch

import numpy

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    use_cuda = True

    writer = SummaryWriter(log_dir='.runs/')

    vector_size = 3000

    # net = VariationalAutoEncoderFC(numpy.linspace(2000, vector_size, (2000 - vector_size) // 100 + 1), channels=1, edge=100)
    img_size = 100 * 100 * 1
    enc = EncoderConv([5, 10, 20], 1, 100, vector_size, last_with_batchnorm=False)
    dec = DecoderConvTranspose([5, 10, 20], 1, 100, vector_size, last_with_batchnorm=False)
    # dec = DecoderConvPad([5, 10, 20], 1, 100, vector_size, last_with_batchnorm=False)
    # enc = EncoderFC([img_size * 1, 4000, vector_size], last_with_batchnorm=False)
    # dec = DecoderFC([vector_size, 5000, img_size], [-1, 1, 100, 100], last_with_batchnorm=False)
    net = AutoEncoder(
        enc,
        dec
    )

    random.seed(2019)

    normalizeInput = NormalizeImageBatch((0.5,), (0.5,))
    trans = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((100, 100)),
                                transforms.RandomChoice([
                                    transforms.RandomHorizontalFlip(1.0),
                                    transforms.RandomVerticalFlip(1.0),
                                    transforms.Compose([
                                        transforms.RandomVerticalFlip(1.0),
                                        transforms.RandomHorizontalFlip(1.0)
                                    ])
                                ]),
                                transforms.ToTensor(),
                                normalizeInput
                                ])
    inputTransform = transforms.Compose([
        ToNormalHeightmap()
    ])

    normalizeOutput = NormalizeImageBatch((-1.0,), (2.0,))

    toImage = transforms.Compose([
        normalizeOutput,
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
    ])
    toImg = transforms.ToPILImage()
    toNormalMap = ToNormalMap()
    toNormalHeightmap = ToNormalHeightmap()

    limit = None
    batch_size = 4 * 2

    train_set, test_set = net.load_dataset(".generated2/", trans, train_split=0.75, shuffle=True, limit=limit)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=3)

    if torch.cuda.is_available() and use_cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, step_size_up=50)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 30)

    train_img_index = 0
    train_img_features = (torch.rand(1, vector_size) * 2 - 1).cuda()

    epoch = 0
    total_loss = 10
    accuracy = 0

    save_every_nth_epoch = 10

    net_name = ".nets/autoencoder"
    #net.load(net_name)

    while accuracy < 0.97:
        # UCZENIE
        net.train(True)
        total_loss = 0
        example_shown = False
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            optimizer.zero_grad()
            if torch.cuda.is_available() and use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            output = net(inputs)

            loss = net.loss(output, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            input_grid = torchvision.utils.make_grid(normalizeOutput(inputs), 4).cpu()
            output_grid = torchvision.utils.make_grid(normalizeOutput(output), 4).cpu()
            #plt.imshow(numpy.vstack((numpy.asarray(input_grid), numpy.asarray(output_grid))))
            #plt.pause(0.0001)

            writer.add_image('Training/Input images', input_grid, epoch)
            writer.add_image('Training/Output images', output_grid, epoch)

        lr = optimizer.param_groups[0]["lr"]
        print('[Epoch: %d] loss: %.10f, LR: %.10f' % (epoch + 1, total_loss, lr))
        epoch += 1
        scheduler.step(total_loss)

        train_loss = total_loss

        if len(test_set) > 0:
            # TESTY
            net.train(False)
            correct = 0
            total = len(test_set)
            total_loss = 0
            with torch.no_grad():
                for (images, labels) in test_loader:

                    if torch.cuda.is_available() and use_cuda:
                        images, labels = images.cuda(), labels.cuda()

                    outputs = net(images)

                    for i in range(0, outputs.shape[0]):

                        image = images[i]
                        output = outputs[i]

                        input_img = toImage(image.cpu()).convert("RGB")
                        output_img = toImage(output.cpu()).convert("RGB")

                        if random.random() < max((1.0 / total), 0.001) and not example_shown:
                            input_img.save(".autoencoder/iterations/epoch{}input.png".format(epoch))
                            output_img.save(".autoencoder/iterations/epoch{}output.png".format(epoch))

                            plt.imshow(numpy.vstack((numpy.asarray(normalize(input_img)), numpy.asarray(normalize(output_img)))), cmap="gray")
                            plt.pause(0.0001)
                            example_shown = True

                        loss = net.loss(output, image)
                        total_loss += loss.item()
                        if loss.item() < 0.01:
                            correct += 1

                if total != 0:
                    accuracy = float(correct / total)
                    print('Accuracy of the network on the test images: %.3f %%, loss: %.10f' % (100 * accuracy, total_loss))

        writer.add_scalars('Loss', {
            'train': train_loss,
            'validation': total_loss
        }, epoch)

        net.train(False)

        train_img = toImage(net.decode(train_img_features)[0].cpu())
        train_img.save(".autoencoder/train_imgs/.train_img_" + str(train_img_index) + ".png", "PNG")
        train_img_index += 1

        if save_every_nth_epoch is not None:
            if save_every_nth_epoch > 0:
                if (epoch % save_every_nth_epoch) == 0 and epoch != 0:
                    net.save(net_name)
                    print("Network saved!")

    net.save(net_name)

    writer.close()

    for i in range(0, 100):
        test_img = toImage(net.decode((torch.rand(1, vector_size) * 2 - 1).cuda())[0].cpu())
        test_img.save("results/autoencoder/.test_img" + str(i) + ".png", "PNG")
