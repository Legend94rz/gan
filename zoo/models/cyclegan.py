from ..common import NLayer, Resnet9block
from argparse import Namespace
from torch import nn
import torch as T
import itertools as it


class CycleGAN:
    def __init__(self, opt: Namespace):
        self.opt = opt
        self.G = Resnet9block().to(opt.dev)
        self.F = Resnet9block().to(opt.dev)
        self.DX = NLayer().to(opt.dev)
        self.DY = NLayer().to(opt.dev)

        self.optimizer_G = T.optim.Adam(it.chain(self.G.parameters(), self.F.parameters()))
        self.optimizer_D = T.optim.Adam(it.chain(self.DX.parameters(), self.DY.parameters()))





