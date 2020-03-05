import argparse
import torch as T
from abc import abstractmethod, ABC


# todo: result save path
class BaseOption(ABC):
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Some Utils For Merging/Outputing Results.")
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,3,2. use -1 for CPU')
        self._parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
        self._parser.add_argument('--epochs', type=int, default=10, help='training epochs.')
        self._parser.add_argument('--ncpu', type=int, default=1, help='cpu for data fetching.')
        self._parser.add_argument('--inference', action='store_true', help='whether inference or not.')
        self._parser.add_argument('--save_path', type=str, default='./outputs', help='folder for results.')
        self.other_option()
        self._parser.parse_args(namespace=self)
        self.post_process()

    @abstractmethod
    def other_option(self):
        raise NotImplementedError()

    def post_process(self):
        self.gpu_ids = [int(x) for x in self.gpu_ids.split(',') if int(x) >= 0]
        self.dev = T.device(f'cuda:{self.gpu_ids[0]}' if T.cuda.is_available() and len(self.gpu_ids) > 0 else 'cpu')
