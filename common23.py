#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch

def get_autoencoder3(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=64, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )


def get_autoencoder7(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2,
                  padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1), #loco20
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=2,
                  padding=3),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1), #loco19
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2,
                  padding=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2,
                  padding=3),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2,
        #           padding=3),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # # decoder
        # nn.Upsample(size=3, mode='bilinear'),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
        #           padding=2),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.2),
        # nn.Upsample(size=8, mode='bilinear'),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
        #           padding=4),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.2),
        # nn.Upsample(size=15, mode='bilinear'),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
        #           padding=4),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
                  padding=4),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
                  padding=4),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
                  padding=4),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=64, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1,
                  padding=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7,
                  stride=1, padding=3)
    )

def get_para_net(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=1 * pad_mult),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                  padding=4 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64)
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        padding_11 = padding - kernel_size//2
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, padding=padding_11)  # 添加1x1卷积
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return self.relu(out1 + out2)


def get_central_net(out_channels=384, padding=True):
    pad_mult = 1 if padding else 0
    layers = [
        ConvBlock(3, 256, kernel_size=3, padding=1 * pad_mult),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=1 * pad_mult),
        ConvBlock(256, 512, kernel_size=3, padding=1 * pad_mult),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        ConvBlock(512, 512, kernel_size=3, padding=1 * pad_mult),
        ConvBlock(512, 512, kernel_size=3, padding=1 * pad_mult),
        ConvBlock(512, out_channels, kernel_size=3, padding=1 * pad_mult),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
    ]

    return nn.Sequential(*layers)


class student(nn.Module):
    def __init__(self, out_channels=384, padding=True):
        super(student, self).__init__()
        self.student2 = get_para_net(out_channels, padding)
        self.student3 = get_central_net(out_channels, padding)

    def forward(self, x):
        out1 = self.student2(x)
        out2 = self.student3(x)
        return torch.cat((out1, out2), dim=1)


def get_student(out_channels=384, padding=True):
    return student(out_channels, padding)


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)