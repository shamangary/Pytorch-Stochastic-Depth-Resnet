# Pytorch-Stochastic-Depth-Resnet
Pytorch Implementation of Deep Networks with Stochastic Depth https://arxiv.org/abs/1603.09382

Original torch implementation: https://github.com/yueatsprograms/Stochastic_Depth

Uniform probability is set to prob=0.5. Note that this project does not provide linear decay probability.

**Speed up resnet training process around 1.66x**

# How to use?
```
from TYY_stodepth import *

net = resnet18_StoDepth(pretrained=True)

net = resnet101_StoDepth(pretrained=True)
```

# Something you should know
The original paper uses the following equation in testing.
```
out = self.prob*out + identity
```
However, I found that sometimes it could cause performance degradation.
Therefore, I split different implemenation in two files:
```
TYY_stodepth.py: corresponding to "out = out + identity"
```
```
TYY_stodepth_ori.py: corresponding to "out = self.prob*out + identity"
```

