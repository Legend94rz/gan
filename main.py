from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt
from torchvision import transforms as trans
import logging
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from zoo.common import BaseOption, dataset_helper as hp
from zoo.models import CycleGANOption, CycleGAN
from pathlib import Path
import logging
import logging.config
import yaml
if Path('log.config').exists():
    with open('log.config', 'r') as file:
        logging.config.dictConfig(yaml.safe_load(file.read()))


if __name__ == "__main__":
    opt = CycleGANOption()
    save_path = Path(opt.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(save_path)

    transform = trans.Compose([trans.ToTensor(), trans.Resize((320, 180))])
    dataset = hp.UnalignedFolder('./dataset/game_img', transform=transform)
    train_ds, val_ds = random_split(dataset, lengths=[int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_loader = DataLoader(train_ds, num_workers=opt.ncpu, batch_size=opt.batch_size)
    test_loader = DataLoader(val_ds, num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam_test = next(iter(test_loader))
    # img_list = []
    # plt.ioff()
    m = CycleGAN(opt)
    if (save_path / opt.pretrained).exists():
        m.load(save_path/opt.pretrained)
    for e in range(opt.epochs):
        sam_train = None
        for i, data in enumerate(train_loader):
            if i == 0:
                sam_train = data
            loss_d, loss_g = m.update(data)
            writer.add_scalars("loss", {'discriminator': loss_d, 'generator': loss_g})
            logging.info(f'Epoch[{e}/{opt.epochs}]  batch[{i}]  LossG[{loss_g}]  LossD[{loss_d}]')

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
