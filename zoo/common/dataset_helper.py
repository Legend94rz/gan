import zipfile as zip
from torch.utils.data import IterableDataset, Dataset
import torch as T
from PIL import Image
from torchvision import transforms as trans
import re
from pathlib import Path


class CelebAFaces(IterableDataset):
    """
    ref page: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
    """
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
        if self.transform is None:
            self.transform = trans.ToTensor()

    def __iter__(self):
        worker_info = T.utils.data.get_worker_info()
        num_proc = 1 if worker_info is None else worker_info.num_workers
        wid = 0 if worker_info is None else worker_info.id
        with zip.ZipFile(self.filepath) as zf:
            lst = zf.namelist()[1:]  # exclude the first one, which is a folder (point to parent)
            for name in lst[wid::num_proc]:
                with zf.open(name) as f:
                    jpg = Image.open(f).convert('RGB')
                yield self.transform(jpg)


class UnalignedZipped(IterableDataset):
    """
    ref page: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    summer2winter_yosemite.zip
    horse2zebra.zip
    """
    def __init__(self, filepath, subdir, split='train', transform=None):
        self.filepath = filepath
        self.subdir = subdir
        self.split = split
        self.transform = transform
        if self.transform is None:
            self.transform = trans.ToTensor()

    def __iter__(self):
        worker_info = T.utils.data.get_worker_info()
        num_proc = 1 if worker_info is None else worker_info.num_workers
        wid = 0 if worker_info is None else worker_info.id
        with zip.ZipFile(self.filepath) as zf:
            lst = zf.namelist()
            lstA = [x for x in lst if re.fullmatch(f'{self.subdir}/{self.split}A/.+', x) is not None]
            lstB = [x for x in lst if re.fullmatch(f'{self.subdir}/{self.split}B/.+', x) is not None]
            for i in range(wid, max(len(lstA), len(lstB)), num_proc):
                with zf.open(lstA[i % len(lstA)]) as fa, zf.open(lstB[i % len(lstB)]) as fb:
                    jpga = Image.open(fa).convert('RGB')
                    jpgb = Image.open(fb).convert('RGB')
                yield self.transform(jpga), self.transform(jpgb)


class UnalignedFolder(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.imgas = list((self.folder / 'A').iterdir())
        self.imgbs = list((self.folder / 'B').iterdir())
        self.transform = transform
        if self.transform is None:
            self.transform = trans.ToTensor()

    def __getitem__(self, i):
        ia = Image.open(self.imgas[i % len(self.imgas)]).convert('RGB')
        ib = Image.open(self.imgbs[i % len(self.imgbs)]).convert('RGB')
        return self.transform(ia), self.transform(ib)

    def __len__(self):
        return max(len(self.imgbs), len(self.imgbs))
