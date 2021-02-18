from torch.utils.data.dataset import Dataset
import torch


class MNISTDatasetWrapper(Dataset):
    def __init__(self, current_dataset, transforms=None, limit=None):
        self.examples = []
        for n, t in enumerate(current_dataset):
            if limit is not None and n > limit:
                break;
            size = t.size()
            self.examples.append(t.to(torch.float32).view(1, size[0], size[1]) / 255.0)
        self.transforms = transforms

    def __getitem__(self, item):
        img = self.examples[item]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img

    def __len__(self):
        return len(self.examples)
