from .base_model import BaseModel
from ..common import DCGenerator, DCDiscriminator, DisLoss, GenLoss, BaseOption
import torch as T


class DCGanOption(BaseOption):
    def other_option(self):
        self._parser.add_argument('--nz', type=int, default=3, help='number of features drawn from dist. randomly.')


class DCGan(BaseModel):
    def __init__(self, opt: BaseOption):
        super().__init__(opt)
        self.G = DCGenerator().to(opt.dev)
        self.D = DCDiscriminator().to(opt.dev)
        self.optimizer_G = T.optim.Adam(self.G.parameters())
        self.optimizer_D = T.optim.Adam(self.D.parameters())
        self.lossD_fn = DisLoss(self.D)
        self.lossG_fn = GenLoss()

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
        noise = T.randn(len(data), self.opt.nz, 1, 1, device=self.opt.dev)
        real_data = data.to(self.opt.dev)

        # update D:
        # tricks: split real and fake in 2 batches, rather than mix them into 1 batch.
        self(noise)
        self.optimizer_D.zero_grad()  # Clears the gradients of all optimized Tensors.
        loss_d = self.lossD_fn(self.fake_data, real_data)
        loss_d.backward()
        self.optimizer_D.step()

        # update G:
        self.optimizer_G.zero_grad()
        loss_g = self.lossG_fn(self.D(self.fake_data))
        loss_g.backward()
        self.optimizer_G.step()
        return loss_d, loss_g
