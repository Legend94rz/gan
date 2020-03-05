from zoo.common import BaseOption, dataset_helper as hp
from zoo.models import CycleGANOption, CycleGAN
import torch as T
from torch.utils.data import DataLoader
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt
from torchvision import transforms as trans
import logging
import sys


def init_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    streamhand = logging.StreamHandler(sys.stdout)
    streamhand.setFormatter(format)
    logger.addHandler(streamhand)
    return logger


if __name__ == "__main__":
    # todo: multi gpu
    logger = init_logger()
    opt = CycleGANOption()
    transform = trans.Compose([trans.Resize(286), trans.CenterCrop(256), trans.ToTensor(),
                               trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = DataLoader(hp.Summer2Winter('./dataset/summer2winter_yosemite.zip', split='train', transform=transform),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam_train = sam_test = next(
        iter(DataLoader(hp.Summer2Winter('./dataset/summer2winter_yosemite.zip', split='test', transform=transform),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)))
    img_list = []
    plt.ioff()
    m = CycleGAN(opt)
    for e in range(opt.epochs):
        for i, data in enumerate(loader):
            if i == 0:
                sam_train = data
            loss_d, loss_g = m.update(data)
            logger.info(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')
            if i % 500 == 0:
                fake_x, fake_y = m(sam_train[0][:64], sam_train[1][:64])
                x2y_in_train = vutil.make_grid(fake_y.detach().cpu(), normalize=True)
                y2x_in_train = vutil.make_grid(fake_x.detach().cpu(), normalize=True)
                fake_x, fake_y = m(sam_test[0][:64], sam_test[1][:64])
                x2y_in_test = vutil.make_grid(fake_y.detach().cpu(), normalize=True)
                y2x_in_test = vutil.make_grid(fake_x.detach().cpu(), normalize=True)
                img = vutil.make_grid([x2y_in_train, y2x_in_train, x2y_in_test, y2x_in_test], nrow=2, padding=8) \
                    .numpy().transpose([1, 2, 0])
                plt.axis('off')
                img_list.append([plt.imshow(img)])
                plt.show()
    plt.ion()
    fig = plt.figure(figsize=(16, 16))
    gif = anim.ArtistAnimation(fig, img_list, interval=50, repeat_delay=1000, blit=True)
    gif.save("result.mp4")
    plt.show()
