from p2t import PatchEmbed
from SCAttn import *
import torch

from RepLKNet import *

# d = torch.randn([3, 4, 5])
# B, hw, chans = d.shape
#
# F1 = CAttn(chans, hw, 3)
# F2 = SAttn(chans, hw, 3)
# y1, o1 = F1(d)
# y2, o2 = F2(d)
#
# print(o1.shape, y1.shape)
# print(o2.shape, y2.shape)
# print('************************************\n')
#
# d1 = torch.randn([3, 4, 5, 6])
# B, mh, hw, chans = d1.shape
#
# F3 = MHAttn(mh)
# y3, o3 = F3(d1)
#
# print(o3.shape, y3.shape)
print('************************************\n')

d2 = torch.randn([4, 3, 224, 224])
B, C, H, W = d2.shape

PM = PatchEmbed(img_size=H, patch_size=4, kernel_size=3, in_chans=C, embed_dim=96, overlap=True)
y4, (h, w) = PM(d2)
_, hw, chans = y4.shape
F3 = SAttn(chans, hw)
F4 = CAttn(chans, hw)
y5 = F3(y4)
y6 = F4(y5)
#print(y4.shape, y5.shape, y6.shape)

x1 = torch.randn([4, 3136, 64])
x2 = torch.randn([4, 784, 128])
x3 = torch.randn([4, 196, 320])
x4 = torch.randn([4, 49, 512])

F5 = FeatureFusion(256)

y = F5(x1, x2, x3, x4)
#print(y.shape)
#print(x1.shape, x2.shape, x3.shape, x4.shape)

x1 = torch.randn([4, 3136, 64]).reshape(4, 56, 56, -1).permute(0, 3, 1, 2)
Rl1 = RepLKBlock(29, 64, 64, prob=0.3)
z1 = Rl1(x1)
print(z1.shape)

x2 = torch.randn([4, 784, 128]).reshape(4, 28, 28, -1).permute(0, 3, 1, 2)
Rl2 = RepLKBlock(15, 128, 128, prob=0.3)
z2 = Rl2(x2)
print(z2.shape)

x3 = torch.randn([4, 196, 320]).reshape(4, 14, 14, -1).permute(0, 3, 1, 2)
Rl3 = RepLKBlock(9, 320, 320, prob=0.3)
z3 = Rl3(x3)
print(z3.shape)

x4 = torch.randn([4, 3, 224, 224])
Rl4 = RepLKBlock(31, 3, 3, prob=0.3)
z4 = Rl4(x4)
print(z4.shape)
