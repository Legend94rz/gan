from ..common import DCGenerator, DCDiscriminator
from argparse import Namespace
from torch import nn
import torch as T


class DCGan:
    nz = 100

    def __init__(self, opt: Namespace):
        # todo: move to base model
        self.opt = opt

        self.G = DCGenerator().to(self.opt.dev)
        self.D = DCDiscriminator().to(self.opt.dev)
        self.optimizer_G = T.optim.Adam(self.G.parameters())
        self.optimizer_D = T.optim.Adam(self.D.parameters())
        self.loss = nn.BCEWithLogitsLoss()

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

        # sample noise
        noise = T.randn(len(data), DCGan.nz, 1, 1, device=self.opt.dev)

        # update D:
        self.optimizer_D.zero_grad()  # Clears the gradients of all optimized Tensors.

        # tricks: split real and fake in 2 batches, rather than mix them into 1 batch.
        # 1. using real samples
        real_data = data.to(self.opt.dev)
        real_pred = self.D(real_data)
        real_label = T.tensor(1.0).expand_as(real_pred)
        loss_real = self.loss(real_pred, real_label)
        loss_real.backward()
        # 2. using noise
        fake_data = self.G(noise)
        fake_pred = self.D(fake_data.detach())      # fix generator by detach. todo: 不用detach行不行？optD不更新G吧？
        fake_label = T.tensor(1.0).expand_as(fake_pred)
        loss_fake = self.loss(fake_pred, fake_label)
        loss_fake.backward()
        # 3. optimize
        self.optimizer_D.step()

        # update G:
        self.optimizer_G.zero_grad()
        pred = self.D(fake_data)
        loss_G = self.loss(pred, real_label)
        loss_G.backward()
        self.optimizer_G.step()
        return (loss_fake+loss_real)/2.0, loss_G

