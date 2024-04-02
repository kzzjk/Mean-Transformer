#!/usr/bin/env python
# coding: utf-8
import torch.nn as nn
# from torchinfo import summary
from timm.models.layers import DropPath


class DepthWiseConv2d(nn.Module):

    def __init__(self, c_in, c_out, kernels_per_layer, kernel_size, stride):
        super(DepthWiseConv2d, self).__init__()

        self.conv_dw = nn.Conv2d(c_in, c_in*kernels_per_layer, kernel_size=kernel_size, padding=1, groups=c_in, bias=False)
        self.conv_pw = nn.Conv2d(c_in*kernels_per_layer, c_out, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):

        return self.conv_pw(self.conv_dw(x))

# dw_layer = DepthWiseConv2d(c_in=3, c_out=16, kernels_per_layer=8).cuda()
# summary(dw_layer, (3, 256, 256), device="cuda")


class Stem(nn.Module):

    def __init__(self, c_in, c_out):
        super(Stem, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, bias=False)
        self.conv_dw1 = DepthWiseConv2d(c_out, c_out, kernels_per_layer=8, stride=1, kernel_size=3)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv_dw2 = DepthWiseConv2d(c_out, c_out, kernels_per_layer=8, stride=2, kernel_size=3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv2(x)
        x = self.conv_dw2(x)

        return x


class Transition(nn.Module):

    def __init__(self, c_in, c_out, kernels_per_layer, kernel_size):
        super(Transition, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=1, stride=1, padding="same", bias=False)
        self.dw_conv = DepthWiseConv2d(c_in, c_out, kernels_per_layer, kernel_size, stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.dw_conv(x)

        return x


# layer = Transition(c_in=3, c_out=16, kernels_per_layer=8, kernel_size=3).cuda()
# summary(layer, (32, 3, 256, 256))
# layer = Stem(c_in=3, c_out=32).cuda()
# summary(layer, (32, 3, 256, 256))


class RepLKBlock(nn.Module):

    def __init__(self, kernel_size, c_in, c_out, prob):
        super(RepLKBlock, self).__init__()
        
        # Only works for kernel sizes up to 9x9 5 1 7 2 9 3 11 4 13 5 15 6 17 7 19 8 21 9 23 10 25 11 27 12 29 13 31 14
        if kernel_size <= 9:
            padding = kernel_size // 3
        elif kernel_size == 31:
            padding = 14
        elif kernel_size == 29:
            padding = 13
        elif kernel_size == 15:
            padding = 6

        self.bn = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False)
        self.conv_dw = DepthWiseConv2d(c_out, c_out, stride=1, kernels_per_layer=8, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=padding, bias=False)
        self.drop_path = DropPath(prob)

    def forward(self, x):

        add = x

        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv_dw(x)
        x = self.conv2(x)

        return self.drop_path(x) + add


class ConvFFN(nn.Module):

    def __init__(self, c_in, c_out, prob):
        super(ConvFFN, self).__init__()

        self.bn = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", stride=1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding="same", stride=1, bias=False)
        self.drop_path = DropPath(prob)

    def forward(self, x):

        add = x

        x = self.bn(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)

        return self.drop_path(x) + add


# block = RepLKBlock(c_in=16, c_out=16, kernel_size=15, prob=0.2).cuda()
# summary(block, (32, 16, 256, 256), device="cuda")
# block = ConvFFN(c_in=16, c_out=16).cuda()
# summary(block, (32, 16, 256, 256), device="cuda")


# In[24]:


# Stem
# Stage 1: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 1
# Stage 2: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 2
# Stage 3: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 3
# Stage 4: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 4

class RepLKNet(nn.Module):

    def __init__(self, c_in, num_classes, num_blocks_per_stage=[2, 2, 18, 2], prob=0.3, kernel_sizes=[31, 29, 15, 7]):
        super(RepLKNet, self).__init__()

        c_out = 32

        self.stem = Stem(c_in, c_out)

        modules1 = []

        for i in range(num_blocks_per_stage[0]):
            modules1.append(RepLKBlock(kernel_sizes[0], c_out, c_out, prob=prob)) 
            modules1.append(ConvFFN(c_out, c_out, prob=prob))
    
        self.stage1 = nn.Sequential(*modules1)
        self.transition1 = Transition(c_out, c_out*2, kernels_per_layer=8, kernel_size=3)
        c_out = c_out*2

        modules2 = []

        for i in range(num_blocks_per_stage[1]):
            modules2.append(RepLKBlock(kernel_sizes[1], c_out, c_out, prob=prob)) 
            modules2.append(ConvFFN(c_out, c_out, prob=prob))

        self.stage2 = nn.Sequential(*modules2)
        self.transition2 = Transition(c_out, c_out*2, kernels_per_layer=16, kernel_size=3)
        c_out = c_out*2

        modules3 = []

        for i in range(num_blocks_per_stage[2]):
            modules3.append(RepLKBlock(kernel_sizes[2], c_out, c_out, prob=prob)) 
            modules3.append(ConvFFN(c_out, c_out, prob=prob))

        self.stage3 = nn.Sequential(*modules3)
        self.transition3 = Transition(c_out, c_out*2, kernels_per_layer=32, kernel_size=3)
        c_out = c_out*2

        modules4 = []

        for i in range(num_blocks_per_stage[3]):
            modules4.append(RepLKBlock(kernel_sizes[3], c_out, c_out, prob=prob)) 
            modules4.append(ConvFFN(c_out, c_out, prob=prob))

        self.stage4 = nn.Sequential(*modules4)
        self.transition4 = Transition(c_out, c_out*2, kernels_per_layer=64, kernel_size=3)
        c_out = c_out*2

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(*[nn.Linear(512, c_out//2),
                                nn.ReLU(inplace=True),
                                nn.Linear(c_out//2, c_out//4),
                                nn.ReLU(inplace=True),
                                nn.Linear(c_out//4, num_classes)])
        
    def forward(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        x = self.transition3(x)
        x = self.stage4(x)
        x = self.transition4(x)
        x = self.adaptive_pool(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x)

        return x

# model = RepLKNet(3, 10).cuda()
# summary(model, (32, 3, 256, 256), kernel_sizes=[31, 29, 15, 7])




