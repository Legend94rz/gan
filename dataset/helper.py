import zipfile as zip
from torch.utils.data import IterableDataset
import torch as T
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
p = Path(__file__).parent


class CelebAFaces(IterableDataset):
    def __iter__(self):
        worker_info = T.utils.data.get_worker_info()
        num_proc = 1 if worker_info is None else worker_info.num_workers
        wid = 0 if worker_info is None else worker_info.id
        with zip.ZipFile(p/'img_align_celeba.zip') as zf:
            lst = zf.namelist()[1:]  # exclude the first one, which is a folder (point to parent)
            for name in lst[wid::num_proc]:
                with zf.open(name) as f:
                    jpg = Image.open(f).convert('RGB')
                    cvjpg = cv2.cvtColor(np.array(jpg), cv2.COLOR_RGB2BGR)
                    yield cvjpg
