from Models.AutoEncoder import DecoderConvTranspose
import torch

from Images.Normalization import NormalizeImageBatch
from torchvision.transforms import transforms

if __name__ == "__main__":
    vector_size = 1000
    decoder = DecoderConvTranspose([9, 16, 25], 1, 100, vector_size, last_with_batchnorm=False)

    decoder.load("../ae_decoder")

    normalizeOutput = NormalizeImageBatch((-1.0,), (2.0,))

    toImage = transforms.Compose([
        normalizeOutput,
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
    ])

    img = toImage(decoder.forward(torch.rand(1, vector_size) * 2 - 1)[0])
    img.save("ae_test.png")