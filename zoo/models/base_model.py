from zoo.common import BaseOption
import torch as T


class BaseModel:
    def __init__(self, opt: BaseOption):
        self.opt = opt

    def save(self, path):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, T.optim.Optimizer) or isinstance(v, T.nn.Module):
                d[k] = v.state_dict()
        T.save(d, path)

    def load(self, path):
        ckpt = T.load(path)
        for k, v in self.__dict__.items():
            if isinstance(v, T.optim.Optimizer) or isinstance(v, T.nn.Module):
                v.load_state_dict(ckpt[k])
