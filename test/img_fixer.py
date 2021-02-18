import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import threading

from Models.BaseModel import BaseModule
from Images.Normalization import ImageToRange

if __name__ == "__main__":
    def save_fixed(_img, _iteration):
        _fixed = _img
        _fixed = _fixed.squeeze(0)
        _fixed = _fixed.cpu()
        _fixedImg = toImage(_fixed)
        _fixedImg.save("fixed_iterations/fixed" + str(_iteration) + ".png", "PNG")
        print("Saved iteration " + str(_iteration))

    net = BaseModule.load("../.nets/frd2")

    trans = transforms.Compose([
        transforms.Resize((64, 64)), transforms.Grayscale(),
        transforms.ToTensor(),
        ImageToRange(0, 1)
    ])
    toImage = transforms.Compose([
        ImageToRange(0, 1),
        transforms.ToPILImage()
    ])

    with Image.open("test.png") as img:
        img = trans(img).float()
        img = Variable(img, requires_grad=True)
        img = img.unsqueeze(0)

        target = torch.tensor([1]).float()

        if torch.cuda.is_available():
            net = net.cuda()
            img = img.cuda()
            target = target.cuda()

        loss = net.loss(net(img)[0], target)

        saveEveryNthIteration = 500

        def img_plot():
            plt.ion()

            image_plot = plt.imshow(np.asarray(toImage(img.squeeze(0).cpu())), cmap="gray")
            while thread_working:
                image_plot.set_data(np.asarray(toImage(img.squeeze(0).cpu())))
                plt.pause(0.001)

        thr = threading.Thread(target=img_plot)
        thread_working = True
        thr.start()

        iteration = 1
        while loss.item() != 0.0:
            answer = net(img)[0]
            loss = net.loss(answer, target)
            img.retain_grad()
            loss.backward()

            if loss.item() != 0:
                img -= 1e-5 * img.grad
            img.data.clamp_(0, 1)

            if (iteration == 1) or (iteration % saveEveryNthIteration == 0):
                save_fixed(img, iteration)
                print('Loss: %.10f' % loss.item())
            iteration +=1
        save_fixed(img, iteration)
        print('Loss: %.10f' % loss.item())
        thread_working = False
