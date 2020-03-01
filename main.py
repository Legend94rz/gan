from zoo.common import BaseOptions, dataset_helper as hp
from zoo.models import DCGan
import torch as T
from torch.utils.data import DataLoader
import torchvision.utils as vutil
from matplotlib import animation as anim
import matplotlib.pyplot as plt
from torchvision import transforms as trans


if __name__ == "__main__":
    opt = BaseOptions().parse()
    loader = DataLoader(hp.Summer2Winter('./dataset/summer2winter_yosemite.zip', split='train'),
                        num_workers=opt.ncpu, batch_size=opt.batch_size)
    sam = T.randn(64, DCGan.nz, 1, 1, device=opt.dev)
    img_list = []
    #plt.ioff()
    #m = DCGan(opt)
    for data in loader:
        print(data)
        pass