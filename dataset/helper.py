import zipfile as zip
from torch.utils.data import IterableDataset
import torch as T
from PIL import Image
import cv2
import numpy as np


class CelebAFaces(IterableDataset):
    def __init__(self, filepath, img_size=(64, 64)):
        self.filepath = filepath
        self.img_size = img_size

    def __iter__(self):
        worker_info = T.utils.data.get_worker_info()
        num_proc = 1 if worker_info is None else worker_info.num_workers
        wid = 0 if worker_info is None else worker_info.id
        with zip.ZipFile(self.filepath) as zf:
            lst = zf.namelist()[1:]  # exclude the first one, which is a folder (point to parent)
            for name in lst[wid::num_proc]:
                with zf.open(name) as f:
                    jpg = Image.open(f).convert('RGB')
                    cvjpg = cv2.cvtColor(np.array(jpg), cv2.COLOR_RGB2BGR)
                    cvjpg = cv2.resize(cvjpg, self.img_size).transpose([2, 0, 1]).astype('float32')
                    yield cvjpg
