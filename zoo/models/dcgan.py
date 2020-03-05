from .base_model import BaseModel
from ..common import DCGenerator, DCDiscriminator, DisLoss, GenLoss, BaseOption
from torch import nn
import torch as T


class DCGan(BaseModel):
    nz = 100

    def __init__(self, opt: BaseOption):
        super().__init__(opt)
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
        # 1. model.forward()
        # 2. optimizer.zero_grad()
        # 3. compute loss
        # 4. loss.backward()
        # 5. optimizer.step()

        # prepare input
        noise = T.randn(len(data), DCGan.nz, 1, 1, device=self.opt.dev)
        real_data = data.to(self.opt.dev)

        # update D:
        # tricks: split real and fake in 2 batches, rather than mix them into 1 batch.
        self(noise)
        self.optimizer_D.zero_grad()  # Clears the gradients of all optimized Tensors.
        loss_d = self.lossD_fn(self.fake_data, real_data)
        loss_d.backward()
        self.optimizer_D.step()

        # update G:
        pred = self.D(self.fake_data)
        self.optimizer_G.zero_grad()
        loss_g = self.lossG_fn(pred)
        loss_g.backward()
        self.optimizer_G.step()
        return loss_d, loss_g

