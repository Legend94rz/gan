from ..common import DCGenerator, DCDiscriminator, DisLoss, GenLoss
from argparse import Namespace
from torch import nn
import torch as T


class DCGan:
    nz = 100

    def __init__(self, opt: Namespace):
        # todo: move to base model
        self.opt = opt

        self.G = DCGenerator().to(opt.dev)
        self.D = DCDiscriminator().to(opt.dev)
        self.optimizer_G = T.optim.Adam(self.G.parameters())
        self.optimizer_D = T.optim.Adam(self.D.parameters())
        self.lossD_fn = DisLoss(self.D).to(opt.dev)
        self.lossG_fn = GenLoss().to(opt.dev)

    def __call__(self, x):
        self.fake_data = self.G(x)
        return self.fake_data

    def update(self, data):
        # typical update routine:
        # 1. optimizer.zero_grad()
        # 2. model.forward()
        # 3. compute loss
        # 4. loss.backward()
        # 5. optimizer.step()

        # prepare input
        noise = T.randn(len(data), DCGan.nz, 1, 1, device=self.opt.dev)
        real_data = data.to(self.opt.dev)

        # update D:
        # tricks: split real and fake in 2 batches, rather than mix them into 1 batch.
        self.optimizer_D.zero_grad()  # Clears the gradients of all optimized Tensors.
        self(noise)
        loss_d = self.lossD_fn(self.fake_data, real_data)
        loss_d.backward()
        self.optimizer_D.step()

        # update G:
        self.optimizer_G.zero_grad()
        pred = self.D(self.fake_data)
        loss_g = self.lossG_fn(pred)
        loss_g.backward()
        self.optimizer_G.step()
        return loss_d, loss_g

