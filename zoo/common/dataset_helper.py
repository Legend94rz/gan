import zipfile as zip
from torch.utils.data import IterableDataset, Dataset
import torch as T
from PIL import Image
from torchvision import transforms as trans
import re


class CelebAFaces(IterableDataset):
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


class Summer2Winter(Dataset):
    def __init__(self, filepath, split='train', transform=None):
        self.filepath = filepath
        self.split = split
        self.transform = transform
        if self.transform is None:
            self.transform = trans.ToTensor()
        self.zf = zip.ZipFile(self.filepath)
        lst = self.zf.namelist()
        self.lstA = [x for x in lst if re.fullmatch(f'summer2winter_yosemite/{split}A/.+', x) is not None]
        self.lstB = [x for x in lst if re.fullmatch(f'summer2winter_yosemite/{split}B/.+', x) is not None]

    def __getitem__(self, i):
        worker_info = T.utils.data.get_worker_info()
        with self.zf.open(self.lstA[i % len(self.lstA)]) as fa, self.zf.open(self.lstB[i % len(self.lstB)]) as fb:
            print("[%d/%d]: %s \t %s" % (worker_info.id, worker_info.num_workers, self.lstA[i % len(self.lstA)], self.lstB[i % len(self.lstB)]))
            jpga = Image.open(fa).convert('RGB')
            jpgb = Image.open(fb).convert('RGB')
        return jpga, jpgb

    def __len__(self):
        return max(len(self.lstA), len(self.lstB))
