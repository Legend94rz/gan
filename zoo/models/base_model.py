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

    def set_requires_grad(self, nets, require_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            assert isinstance(net, T.nn.Module)
            for p in net.parameters():
                p.requires_grad_(require_grad)
