from .base_model import BaseModel
from ..common import NLayerDiscriminator, ResnetGenerator, GenLoss
from torch import nn
import torch as T
import itertools as it
from zoo.common import DisLoss
from ..common.options import BaseOption


class CycleGANOption(BaseOption):
    def other_option(self):
        self._parser.add_argument('--lamda', type=float, default=0.5, help='weights of cycle loss.')
        self._parser.add_argument('--domx_nc', type=int, default=3, help='X domain channel. 3 for RGB, 1 for grayscale')
        self._parser.add_argument('--domy_nc', type=int, default=3, help='Y domain channel. 3 for RGB, 1 for grayscale')
        self._parser.add_argument('--n_blocks', type=int, default=9, help='resnet blocks used for generators.')


class CycleGAN(BaseModel):
    def __init__(self, opt: BaseOption):
        super().__init__(opt)
        self.G2X = nn.DataParallel(ResnetGenerator(opt.domy_nc, opt.domx_nc, n_blocks=opt.n_blocks), opt.gpu_ids)
        self.G2Y = nn.DataParallel(ResnetGenerator(opt.domx_nc, opt.domy_nc, n_blocks=opt.n_blocks), opt.gpu_ids)
        self.D4X = nn.DataParallel(NLayerDiscriminator(opt.domx_nc), opt.gpu_ids)
        self.D4Y = nn.DataParallel(NLayerDiscriminator(opt.domy_nc), opt.gpu_ids)

        self.optimizer_G = T.optim.Adam(it.chain(self.G2Y.parameters(), self.G2X.parameters()))
        self.optimizer_D = T.optim.Adam(it.chain(self.D4X.parameters(), self.D4Y.parameters()))
        self.d4y_lossfn = DisLoss(self.D4Y)  # no params, thus doesn't have to `to(dev)`
        self.d4x_lossfn = DisLoss(self.D4X)
        self.g2x_lossfn = GenLoss()
        self.g2y_lossfn = GenLoss()
        self.cyc_lossfn = nn.L1Loss()

    def __call__(self, dom_x, dom_y):
        dom_x, dom_y = dom_x.to(self.opt.dev), dom_y.to(self.opt.dev)
        self.fake_x = self.G2X(dom_x)
        self.fake_y = self.G2Y(dom_y)
        self.rec_x = self.G2X(self.fake_y)
        self.rec_y = self.G2Y(self.fake_x)
        return self.fake_x, self.fake_y

    def update(self, data):
        dom_x, dom_y = data[0].to(self.opt.dev), data[1].to(self.opt.dev)
        self(dom_x, dom_y)

        self.optimizer_D.zero_grad()
        loss_dy = self.d4y_lossfn(self.fake_y, dom_y)
        loss_dx = self.d4x_lossfn(self.fake_x, dom_x)
        loss_d = loss_dx + loss_dy
        loss_d.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        loss_gx = self.g2x_lossfn(self.D4X(self.fake_x))
        loss_gy = self.g2y_lossfn(self.D4Y(self.fake_y))
        cyc_x = self.cyc_lossfn(dom_x, self.rec_x)
        cyc_y = self.cyc_lossfn(dom_y, self.rec_y)
        loss_g = loss_gx + loss_gy + self.opt.lamda*(cyc_x+cyc_y)
        # RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed.
        #  Specify retain_graph=True when calling backward the first time.
        loss_g.backward()
        self.optimizer_G.step()

        return loss_d, loss_g
