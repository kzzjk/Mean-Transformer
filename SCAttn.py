import torch
import torch.nn as nn


class CAttn(nn.Module):
    def __init__(self, chans, hw):
        super().__init__()

        self.chans = chans
        self.hw = hw

        self.avg_pool_c = nn.AdaptiveAvgPool2d((1, chans))
        self.max_pool_c = nn.AdaptiveMaxPool2d((1, chans))

        self.conv = nn.Linear(in_features=chans, out_features=chans, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool_c(x)
        y2 = self.max_pool_c(x)
        out = y1 + y2

        out = self.conv(out)
        out = self.sigmoid(out)
        # print(out.shape)

        y = x*out.expand_as(x)

        return y


class SAttn(nn.Module):
    def __init__(self, chans, hw):
        super().__init__()

        self.chans = chans
        self.hw = hw

        self.avg_pool_s = nn.AdaptiveAvgPool2d((hw, 1))
        self.max_pool_s = nn.AdaptiveMaxPool2d((hw, 1))

        self.conv = nn.Linear(in_features=hw, out_features=hw, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool_s(x)
        y2 = self.max_pool_s(x)
        out = y1 + y2

        out = out.transpose(1, 2)
        out = self.conv(out)
        out = out.transpose(1, 2)
        out = self.sigmoid(out)
        # print(out.shape)

        y = x * out.expand_as(x)

        return y


class MHAttn(nn.Module):
    def __init__(self, mh):
        super().__init__()
        self.mh = mh

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Conv1d(in_channels=mh, out_channels=mh, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, nh, hw, c = x.shape
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        out = y1 + y2

        out = out.flatten(2)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = out.reshape(b, nh, 1, -1)

        y = x * out.expand_as(x)

        return y


class FeatureFusion(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.chans = chans
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        b, _, _ = x1.shape
        x1 = x1.reshape(b, -1, self.chans)
        x2 = x2.reshape(b, -1, self.chans)
        x3 = x3.reshape(b, -1, self.chans)
        x4 = x4.reshape(b, -1, self.chans)

        out = torch.cat((x1, x2, x3, x4), dim=1)

        # out = self.sigmoid(x)

        return out
