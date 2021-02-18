import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class BaseModule(nn.Module, ABC):
    def __init__(self):
        super(BaseModule, self).__init__()

    def save(self, name):
        to_save = self
        torch.save(to_save, name + ".net")

    @staticmethod
    def load(name):
        try:
            model = torch.load(name + ".net")
            model.eval()
        except RuntimeError as e:
            print(e)
            return None
        except FileNotFoundError:
            print("File not found. Can't load network")
            return None
        return model

    def load_state(self, name):
        try:
            model = torch.load(name + ".net")
            self.load_state_dict(model.state_dict())
            self.eval()
        except RuntimeError as e:
            print(e)
            return False
        except FileNotFoundError:
            print("File not found. Can't load network")
            return False
        return True

    def get_activations(self, *inputs):
        pass
