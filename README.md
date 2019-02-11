# Pytorch-Stochastic-Depth-Resnet
Pytorch Implementation of Deep Networks with Stochastic Depth https://arxiv.org/abs/1603.09382

Original torch implementation: https://github.com/yueatsprograms/Stochastic_Depth

Uniform probability is set to prob=0.5.

**Speed up resnet training process around 1.66x**

# How to use?
```
# For linear decay probability
from TYY_stodepth_lineardecay import *

net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,0.5])


# For uniform probability
from TYY_stodepth import *

net = resnet18_StoDepth(pretrained=True)

net = resnet101_StoDepth(pretrained=True)

```

# Something you should know for uniform probability
The original paper uses the following equation in testing.
```
out = self.prob*out + identity
```
However, I found that sometimes it could cause performance degradation.


Therefore, I split the different implemenations into two files:
```
TYY_stodepth.py: corresponding to "out = out + identity"
```
```
TYY_stodepth_ori.py: corresponding to "out = self.prob*out + identity"
```

