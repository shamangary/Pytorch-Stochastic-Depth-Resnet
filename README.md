# Pytorch-Stochastic-Depth-Resnet
Pytorch Implementation of Deep Networks with Stochastic Depth https://arxiv.org/abs/1603.09382

Original torch implementation: https://github.com/yueatsprograms/Stochastic_Depth

**Speed up resnet training process around 1.66x**

# How to use?
```
# For linear decay probability
from TYY_stodepth_lineardecay import *

# [testing]: out = self.prob*out + identity
net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,0.5], multFlag=True) 

# [testing]: out = out + identity
net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,0.5], multFlag=False) 

#----------------------------------------------------------------------------------
# For uniform probability
from TYY_stodepth import *

# [testing]: out = self.prob*out + identity
net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[0.5,0.5], multFlag=True)

# [testing]: out = out + identity
net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[0.5,0.5], multFlag=False)
```

# Something you should know
The original paper uses the following equation in testing.
```
out = self.prob*out + identity
```
However, I found that sometimes it could cause performance degradation.

Change "multFlag" to False if you dont want to multiply probability on the testing output.
