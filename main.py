from zoo.common import BaseOptions
from zoo.models import CycleGAN, DCGan
from dataset import helper as hp
import torch as T
from torch.utils.data import DataLoader
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt
from torchvision import transforms as trans


if __name__ == "__main__":
    opt = BaseOptions().parse()
    m = DCGan(opt)
    loader = DataLoader(hp.CelebAFaces('./dataset/img_align_celeba.zip',
                                       transform=trans.Compose([trans.Resize((64, 64)),
                                                                trans.ToTensor(),
                                                                trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam = T.randn(64, DCGan.nz, 1, 1, device=opt.dev)
    img_list = []
    plt.ioff()
    for e in range(opt.epochs):
        for i, data in enumerate(loader):
            loss_d, loss_g = m.update(data)
            # todo: forward, show sample, update return results
            if i % 100 == 0:
                print(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')
            if i % 500 == 0:
                gen_img = vutil.make_grid(m(sam).detach().cpu(), normalize=True).numpy().transpose([1, 2, 0])
                plt.axis('off')
                img_list.append([plt.imshow(gen_img)])
                plt.show()
    plt.ion()
    plt.axis('off')
    fig = plt.figure(figsize=(8, 8))
    gif = anim.ArtistAnimation(fig, img_list, interval=50, repeat_delay=1000, blit=True)
    plt.show()
