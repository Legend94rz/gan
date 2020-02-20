import argparse
import torch as T


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Some Utils For Merging/Outputing Results.")
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
        self.parser.add_argument('--epochs', type=int, default=10, help='training epochs.')
        self.parser.add_argument('--ncpu', type=int, default=1, help='cpu for data fetching.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.dev = T.device('cuda' if T.cuda.is_available() and opt.gpu_ids!='-1' else 'cpu')
        return opt
