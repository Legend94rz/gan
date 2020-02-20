from ..common import NLayer, Resnet9block
from argparse import Namespace
from torch import nn
import torch as T


class CycleGAN:
    def __init__(self, opt: Namespace):
        self.G = Resnet9block()
        self.F = Resnet9block()
        self.DX = NLayer()
        self.DY = NLayer()

        self.optimizer_G = T.optim.Adam([self.G.parameters(), self.F.parameters()])



