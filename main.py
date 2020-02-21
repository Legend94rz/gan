from zoo.common import BaseOptions
from zoo.models import CycleGAN, DCGan
from dataset import helper as hp
import torch as T
from torch.utils.data import DataLoader
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt

if __name__ == "__main__":
    opt = BaseOptions().parse()
    m = DCGan(opt)
    loader = DataLoader(hp.CelebAFaces('./dataset/img_align_celeba.zip'), num_workers=opt.ncpu)
    sam = T.randn(64, DCGan.nz, 1, 1, device=opt.dev)
    img_list = []
    plt.ioff()
    plt.axis('off')
    for e in range(opt.epochs):
        for i, data in enumerate(loader):
            loss_d, loss_g = m.update(data)
            # todo: forward, show sample, update return results
            if i % 100 == 0:
                print(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')
            if i % 500 == 0:
                gen_img = vutil.make_grid(m(sam).detach().cpu(), normalize=True).numpy().transpose([1, 2, 0])
                img_list.append(gen_img)
                plt.imshow(gen_img)
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    gif = anim.ArtistAnimation(fig, img_list, interval=50, repeat_delay=1000, blit=True)
    plt.show()
