import numpy as np
from timm.models.layers.activations import sigmoid
import torch
from torch import nn
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LayerNorm(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)   #.
        y_2 = self.max_pool(x)
        y = (y + y_2)
        y = y.view(y.size(0), -1)
        y = self.fc(y).view(b,c,1,1)
        return y
        # return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        # self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        # self.conv=nn.Conv2d(1, 1, kernel_size=1, stride = 1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        # self.re = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        # result=torch.cat([max_result,avg_result],1)
        result = max_result + avg_result
        output=self.bn(result)
        # output = self.re(output)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,16,7,7)

    cbam = CBAMBlock(channel=16,reduction=16,kernel_size=7)
    output=cbam(input)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in cbam.parameters())))
    print(output.shape)

    