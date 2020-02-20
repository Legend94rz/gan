from torch import nn


class Resnet9block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class NLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 100
        out_channels = 3
        # H_out = (H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # H_out = H_in×stride[0]+(kernel_size[0]-stride[0]−2×padding[0])
        self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(512), nn.ReLU(),  # output: C x 4 x 4. supposing input C x 1 x 1
                                   nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256), nn.ReLU(),  # output: C x 8 x 8
                                   nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(128), nn.ReLU(),  # output: C x 16 x 16
                                   nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(),  # output: C x 32 x 32
                                   nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.Tanh())  # output: C x 64 x 64

    def forward(self, x):
        return self.main(x)


class DCDiscriminator(nn.Module):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        # H_out = int( [H_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1]/stride[0] + 1 )
        # H_out = int( [H_in+2×padding[0]-kernel_size[0]] / stride[0] + 1)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),  # output: C x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def forward(self, y_pred, y_true):
        pass












