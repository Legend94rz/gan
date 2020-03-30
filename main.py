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

    transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader = DataLoader(
        hp.Unaligned('./dataset/horse2zebra.zip', 'horse2zebra', split='train', transform=transform),
        num_workers=opt.ncpu, batch_size=opt.batch_size)

    test_loader = DataLoader(
        hp.Unaligned('./dataset/horse2zebra.zip', 'horse2zebra', split='test', transform=transform),
        num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam_test = next(iter(test_loader))
    # img_list = []
    # plt.ioff()
    m = CycleGAN(opt)
    for e in range(opt.epochs):
        sam_train = None
        for i, data in enumerate(train_loader):
            if i == 0:
                sam_train = data
            loss_d, loss_g = m.update(data)
            writer.add_scalars("loss", {'discriminator': loss_d, 'generator': loss_g})
            logger.info(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')

        fake_x, fake_y = m(sam_train[0][:4], sam_train[1][:4])
        lst_train = [*sam_train[0][:4], *fake_x.detach().cpu(), *sam_train[1][:4], *fake_y.detach().cpu()]
        fake_x, fake_y = m(sam_test[0][:4], sam_test[1][:4])
        lst_test = [*sam_test[0][:4], *fake_x.detach().cpu(), *sam_test[1][:4], *fake_y.detach().cpu()]
        img = vutil.make_grid(lst_train + lst_test, nrow=8).numpy()
        writer.add_image("generated", img, e)
        m.save(save_path/f"cyclegan.pt")
    # plt.ion()
    # fig = plt.figure(figsize=(16, 16))
    # gif = anim.ArtistAnimation(fig, img_list, interval=50, repeat_delay=1000, blit=True)
    # gif.save(save_path/"result.mp4")
    # plt.show()
