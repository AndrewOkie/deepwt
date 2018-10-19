import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dilation):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation

        self.increase_dim = self.in_channels != self.out_channels or self.stride > 1

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=self.dilation, dilation=self.dilation)

        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=self.dilation, dilation=self.dilation)

        if self.increase_dim:
            self.conv_shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride)

    def forward(self, x):
        bn_relu = F.relu(self.bn1(x))
        residual = self.conv1(bn_relu)
        residual = self.conv2(F.relu(self.bn2(residual)))

        if self.increase_dim:
            x = self.conv_shortcut(bn_relu)

        return x + residual


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.res_block = ResBlock(self.in_channels, self.out_channels, stride=2, dilation=self.dilation)

    def forward(self, x):
        return self.res_block(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.res_block = ResBlock(self.in_channels, self.out_channels, stride=1, dilation=self.dilation)
        self.upsample = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x_down, x_up):
        x = self.res_block(x_up)
        x = self.upsample(F.relu(x))
        x = x + x_down
        return x


class UniversalNet(nn.Module):

    def __init__(self, in_channels, out_channels, in_planes=32, depth=3):
        super(UniversalNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_planes = in_planes
        self.depth = depth

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_planes, kernel_size=3, padding=1),
            ResBlock(self.in_planes, self.in_planes, stride=1, dilation=1)
        )

        #self.down_blocks = nn.ModuleList([DownBlock(2 ** idx * self.in_planes, 2 ** (idx + 1) * self.in_planes, dilation=2 ** (idx + 1)) for idx in range(self.depth)])
        self.down_blocks = nn.ModuleList([DownBlock(2 ** idx * self.in_planes, 2 ** (idx + 1) * self.in_planes, dilation=1) for idx in range(self.depth)])

        #self.bottom_block = ResBlock(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, stride=1, dilation=2 ** (self.depth + 1))
        self.bottom_block = ResBlock(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, stride=1, dilation=1)
        #self.bottom_block = nn.Conv2d(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, kernel_size=1)

        #self.up_blocks = nn.ModuleList([UpBlock(2 ** (idx + 1) * self.in_planes, 2 ** idx * self.in_planes, dilation=2 ** (idx + 1)) for idx in reversed(range(self.depth))])
        self.up_blocks = nn.ModuleList([UpBlock(2 ** (idx + 1) * self.in_planes, 2 ** idx * self.in_planes, dilation=1) for idx in reversed(range(self.depth))])

        self.out_conv = nn.Sequential(
            ResBlock(self.in_planes, self.in_planes, stride=1, dilation=1),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, self.out_channels, kernel_size=1)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.in_conv(x)

        encoder = []
        for idx in range(self.depth):
            encoder.append(x)
            x = self.down_blocks[idx](x)

        x = self.bottom_block(x)

        for idx in range(self.depth):
            x = self.up_blocks[idx](encoder[self.depth - 1 - idx], x)

        x = self.out_conv(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SegmentationNet(nn.Module):

    def __init__(self, in_channels, out_channels, in_planes=32, depth=3):
        super(SegmentationNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_planes = in_planes
        self.depth = depth

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_planes, kernel_size=3, padding=1),
            ResBlock(self.in_planes, self.in_planes, stride=1, dilation=1)
        )

        self.down_blocks = nn.ModuleList([DownBlock(2 ** idx * self.in_planes, 2 ** (idx + 1) * self.in_planes, dilation=2 ** (idx + 1)) for idx in range(self.depth)])
        #self.down_blocks = nn.ModuleList([DownBlock(2 ** idx * self.in_planes, 2 ** (idx + 1) * self.in_planes, dilation=1) for idx in range(self.depth)])

        self.bottom_block = ResBlock(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, stride=1, dilation=2 ** (self.depth + 1))
        #self.bottom_block = ResBlock(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, stride=1, dilation=1)
        #self.bottom_block = nn.Conv2d(2 ** self.depth * self.in_planes, 2 ** self.depth * self.in_planes, kernel_size=1)

        self.up_blocks = nn.ModuleList([UpBlock(2 ** (idx + 1) * self.in_planes, 2 ** idx * self.in_planes, dilation=2 ** (idx + 1)) for idx in reversed(range(self.depth))])
        #self.up_blocks = nn.ModuleList([UpBlock(2 ** (idx + 1) * self.in_planes, 2 ** idx * self.in_planes, dilation=1) for idx in reversed(range(self.depth))])

        self.out_conv = nn.Sequential(
            ResBlock(self.in_planes, self.in_planes, stride=1, dilation=1),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, self.out_channels, kernel_size=1)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.in_conv(x)

        encoder = []
        for idx in range(self.depth):
            encoder.append(x)
            x = self.down_blocks[idx](x)

        x = self.bottom_block(x)

        for idx in range(self.depth):
            x = self.up_blocks[idx](encoder[self.depth - 1 - idx], x)

        x = self.out_conv(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
