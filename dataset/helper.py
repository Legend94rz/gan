import zipfile as zip
from torch.utils.data import IterableDataset
import torch as T
from PIL import Image
from torchvision import transforms as trans


class CelebAFaces(IterableDataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        if transform is not None:
            self.transform = transform
        else:
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
