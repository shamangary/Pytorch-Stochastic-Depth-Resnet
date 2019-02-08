# Pytorch-Stochastic-Depth-Resnet
Pytorch Implementation of Deep Networks with Stochastic Depth https://arxiv.org/abs/1603.09382

Original torch implementation: https://github.com/yueatsprograms/Stochastic_Depth

Uniform probability is set to prob=0.5.

**Speed up resnet training process around 1.66x**

# How to use?
```
from TYY_stodepth import *

net = resnet18_StoDepth(pretrained=True)

net = resnet101_StoDepth(pretrained=True)
```
