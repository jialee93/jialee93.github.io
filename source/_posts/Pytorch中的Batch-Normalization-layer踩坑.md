---
title: Pytorch中的Batch Normalization layer踩坑
description: '解惑Pytorch中BN层的原理与使用'
date: 2020-02-03 17:59:17
tags:
	- pytorch
	- 机器学习
	- 深度学习
categories: 科研
	
---

## 1. 注意momentum的定义

Pytorch中的BN层的动量平滑和常见的动量法计算方式是相反的，默认的momentum=0.1
$$
\hat{x}_{\text { new }}=(1-\text { momentum }) \times \hat{x}+\text { momemtum } \times x_{t}
$$
BN层里的表达式为：
$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$
其中*γ*和*β*是可以学习的参数。在Pytorch中，BN层的类的参数有：

```python
CLASS torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

每个参数具体含义参见文档，需要注意的是，affine定义了BN层的参数*γ*和*β*是否是可学习的(不可学习默认是常数1和0). 

## 2. 注意BN层中含有统计数据数值，即均值和方差

**track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `True`

在训练过程中model.train()，train过程的BN的统计数值—均值和方差是**通过当前batch数据估计的**。

并且测试时，model.eval()后，若track_running_stats=True，模型此刻所使用的统计数据是Running status 中的，即通过指数衰减规则，积累到当前的数值。否则依然使用基于当前batch数据的估计值。


## 3. BN层的统计数据更新是在每一次训练阶段model.train()后的forward()方法中自动实现的，**而不是**在梯度计算与反向传播中更新optim.step()中完成

## 4. 冻结BN及其统计数据

从上面的分析可以看出来，正确的冻结BN的方式是在模型训练时，把BN单独挑出来，重新设置其状态为eval (在model.train()之后覆盖training状态）.

解决方案：[转载自](https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385)

> You should use apply instead of searching its children, while named_children() doesn’t iteratively search submodules.

```python
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

model.apply(set_bn_eval)
```

或者，重写module中的train()方法：[转载自](https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8)

```python
def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(MyNet, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
```

## 5. Fix/frozen Batch Norm when training may lead to RuntimeError: expected scalar type Half but found Float 

解决办法：[转载自](https://github.com/NVIDIA/apex/issues/122)

```python
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from apex.fp16_utils import *

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

model = models.resnet50(pretrained=True)
model.cuda()
model = network_to_half(model)
model.train()
model.apply(fix_bn) # fix batchnorm
input = Variable(torch.FloatTensor(8, 3, 224, 224).cuda().half())
output = model(input)
output_mean = torch.mean(output)
output_mean.backward()
```

> Please do
>
> ```
> def fix_bn(m):
>  classname = m.__class__.__name__
>  if classname.find('BatchNorm') != -1:
>      m.eval().half()
> ```
>
> Reason for this is, for regular training it is better (performance-wise) <font color="#dd0000">to **use cudnn batch norm, which requires its weights to be in fp32**, thus batch norm modules are not converted to half in `network_to_half`. However, cudnn does not support batchnorm backward in the eval mode</font> , which is what you are doing, and to use pytorch implementation for this, weights have to be of the same type as inputs.