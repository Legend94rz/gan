from zoo.common import BaseOption, dataset_helper as hp
from zoo.models import CycleGANOption, CycleGAN
from torch.utils.data import DataLoader
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt
from torchvision import transforms as trans
import logging
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from pathlib import Path


def init_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    streamhand = logging.StreamHandler(sys.stdout)
    streamhand.setFormatter(format)
    logger.addHandler(streamhand)
    return logger


if __name__ == "__main__":
    opt = CycleGANOption()
    os.makedirs(opt.save_path, exist_ok=True)
    save_path = Path(opt.save_path)
    logger = init_logger()
    writer = SummaryWriter(save_path)

    transform = trans.Compose([trans.Resize(286), trans.CenterCrop(256), trans.ToTensor(),
                               trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = DataLoader(hp.Summer2Winter('./dataset/summer2winter_yosemite.zip', split='train', transform=transform),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam_train = sam_test = next(
        iter(DataLoader(hp.Summer2Winter('./dataset/summer2winter_yosemite.zip', split='test', transform=transform),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)))
    #img_list = []
    #plt.ioff()
    m = CycleGAN(opt)
    for e in range(opt.epochs):
        for i, data in enumerate(loader):
            if i == 0:
                sam_train = data
            loss_d, loss_g = m.update(data)
            writer.add_scalars("loss", {'discriminator': loss_d, 'generator':loss_g})
            logger.info(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')

        fake_x, fake_y = m(sam_train[0][:4], sam_train[1][:4])
        x2y_in_train = vutil.make_grid(fake_y.detach().cpu(), normalize=True, nrow=2)
        y2x_in_train = vutil.make_grid(fake_x.detach().cpu(), normalize=True, nrow=2)

        fake_x, fake_y = m(sam_test[0][:4], sam_test[1][:4])
        x2y_in_test = vutil.make_grid(fake_y.detach().cpu(), normalize=True, nrow=2)
        y2x_in_test = vutil.make_grid(fake_x.detach().cpu(), normalize=True, nrow=2)
        img = vutil.make_grid([x2y_in_train, y2x_in_train, x2y_in_test, y2x_in_test], nrow=2, padding=8) \
            .numpy().transpose([1, 2, 0])
        writer.add_image("generated", img, e)
        m.save(save_path/f"cyclegan.pt")
    #plt.ion()
    #fig = plt.figure(figsize=(16, 16))
    #gif = anim.ArtistAnimation(fig, img_list, interval=50, repeat_delay=1000, blit=True)
    #gif.save(save_path/"result.mp4")
    #plt.show()
