import random

import torch
import torch.optim as optim
from torchvision import transforms

from Models.FakeRealDiscriminator import FRD

if __name__ == "__main__":

    use_cuda = True
    load_network = False

    vector_size = 2000

    random.seed(2019)

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
                               # transforms.Lambda(lambda x: to_normal_heightmap(x)),
                               # transforms.Lambda(lambda x: normalize(x)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               ])
    toImage = transforms.Compose([
        transforms.Normalize((-1.0,), (2.0,)),
        transforms.ToPILImage(),
        # transforms.Lambda(lambda x: to_heightmap(x)),
        transforms.Resize((100, 100)),
    ])

    limit = None
    batch_size = 4 * 4

    net = FRD()

    train_set, test_set = net.load_dataset("dataset_images/generated/", "dataset_images/fake/", trans, train_split=0.75, limit=limit)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True,
                                               num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False,
                                              num_workers=3)

    if torch.cuda.is_available() and use_cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)

    epoch = 0
    total_loss = 10
    accuracy = 0

    save_every_nth_epoch = 5

    net_name = "frd"
    if load_network:
        net.load(net_name)

    while accuracy < 0.97:
        # UCZENIE
        net.train(True)
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available() and use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            output = net(inputs)

            loss = net.loss(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        lr = optimizer.param_groups[0]["lr"]
        print('[Epoch: %d] loss: %.10f, LR: %.10f' % (epoch + 1, total_loss, lr))
        epoch += 1
        scheduler.step(total_loss)

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

                        loss = net.loss(output, image)
                        total_loss += loss.item()
                        if loss.item() < 0.05:
                            correct += 1

                if total != 0:
                    accuracy = float(correct / total)
                    print('Accuracy of the network on the test images: %.3f %%, loss: %.10f' % (100 * accuracy, total_loss))

        net.train(False)

        if save_every_nth_epoch is not None:
            if save_every_nth_epoch > 0:
                if (epoch % save_every_nth_epoch) == 0 and epoch != 0:
                    net.save(net_name)
                    print("Network saved!")

    net.save(net_name)