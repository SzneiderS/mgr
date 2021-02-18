from torch.utils.data.dataset import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, transforms=None):
        self.examples = []
        self.transforms = transforms

    def __getitem__(self, item):
        img, target = self.examples[item]
        img = Image.open(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.examples)
