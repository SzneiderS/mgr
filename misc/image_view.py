from os import listdir
from os.path import isfile, join
from PIL import Image

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

from Images.QuantizeTransforms import QuantizeImageN2MColors
from Images.GaussianBlurTransform import GaussianBlurPixels
from Images.HeightMapTransforms import ToNormalMap
from Images.shared import *

import torch.nn.functional as F

imageDir = "/sdb7/seb/.generated3/"
images = [f for f in listdir(imageDir) if isfile(join(imageDir, f))]

quantize_transform = QuantizeImageN2MColors(7)
gaussian_blur = GaussianBlurPixels(15)
tonormalmap = ToNormalMap()
grayscale = transforms.Grayscale()

for image in images:
    with Image.open(imageDir + image) as img:
        plt.ion()

        #plt.imshow(np.hstack((np.asarray(normal_map_torch(img)), np.asarray(img))))
        #plt.imshow(np.hstack((np.asarray(normal_map(img)), np.asarray(normal_map_torch(img)))))
        #plt.imshow((np.asarray(normal_map_torch(img))) / 255.0)
        #normalmap = to_normal_heightmap((img))
        #plt.imshow(np.asarray(normalmap))
        #plt.imshow(np.hstack((np.asarray(img), np.asarray(to_heightmap(normalmap)))))
        #plt.imshow(np.asarray(to_heightmap(normalmap)))
        quantized = quantize_transform(img)
        blurred = gaussian_blur(img)
        quantized_blurred = gaussian_blur(quantize_transform(img))
        blurred_quantized = quantize_transform(gaussian_blur(img))
        # plt.imshow(np.hstack((
        #     np.asarray(img),
        #     np.asarray(transformed)
        # )))
        img.save(".original.png")
        quantized.save(".quantized.png")
        blurred.save(".blurred.png")
        quantized_blurred.save(".quantized_blurred.png")
        blurred_quantized.save(".blurred_quantized.png")
        plt.pause(0.001)
        input("Press Enter to continue...")
        plt.close()