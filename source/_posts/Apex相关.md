---
title: Apex相关（同时涉及egg-info的说明）
description: '简述Apex的安装使用以及pip install本地安装'
date: 2020-02-12 19:08:18
tags:
	- pytorch
	- python
	- 环境配置
	- 编译
categories: 科研
---

## What is the difference between FusedAdam optimizer in Nvidia AMP package with the Adam optimizer in Pytorch? 

[摘录自](https://discuss.pytorch.org/t/fusedadam-optimizer-in-nvidia-amp-package/47544)

The Adam optimizer in Pytorch (like all Pytorch optimizers) carries out optimizer.step() by looping over parameters, and launching a series of kernels for each parameter. This can require hundreds of small launches that are mostly bound by CPU-side Python looping and kernel launch overhead, resulting in poor device utilization. Currently, the FusedAdam implementation in Apex flattens the parameters for the optimization step, then carries out the optimization step itself via a fused kernel that combines all the Adam operations. In this way, the loop over parameters as well as the internal series of Adam operations for each parameter are fused such that optimizer.step() requires only a few kernel launches.

The current implementation (in Apex master) is brittle and only works with Amp opt\_level O2\. I’ve got a WIP branch to make it work for any opt\_level (<https://github.com/NVIDIA/apex/pull/351>). I recommend waiting until this is merged then trying it.

## How to use Tensor Cores

[摘录自](https://github.com/NVIDIA/apex/issues/221)

**Convolutions:**
For cudnn versions 7.2 and ealier, @vaibhav0195 is correct: input channels, output channels, and batch size should be multiples of 8 to use tensor cores. However, this requirement is lifted for cudnn versions 7.3 and later. For cudnn 7.3 and later, you don't need to worry about making your channels/batch size multiples of 8 to enable Tensor Core use.

**GEMMs (fully connected layers):**
For matrix A x matrix B, where A has size [I, J] and B has size [J, K], I, J, and K must be multiples of 8 to use Tensor Cores. This requirement exists for all cublas and cudnn versions. This means that for bare fully connected layers, the batch size, input features, and output features must be multiples of 8, and for RNNs, you usually (but not always, it can be architecture-dependent depending on what you use for encoder/decoder) need to have batch size, hidden size, embedding size, and dictionary size as multiples of 8.

**It may also help to set torch.backends.cudnn.benchmark=True**
at the top of your script, which enables pytorch‘s autotuner. Each time pytorch encounters a new set of convolution parameters, it will test all available cudnn algorithms to find the fastest one, then cache that choice to reuse whenever it encounters the same set of convolution parameters again. The first iteration of your network will be slower as pytorch tests all the cudnn algorithms for each convolution, but the second iteration and later iterations will likely be faster.

## FP16半精度带来的精度误差

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190911164622328.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW9qaWFqaWEwMDc=,size_16,color_FFFFFF,t_70)

## Install Nvidia Apex 

若第一次安装需要把项目从github克隆到本地

### Clean the old install before rebuilding:

> pip uninstall apex
> cd apex\_repo\_dir
> rm -rf build (if present)
> rm -rf apex.egg-info (if present)

### Install package：

注：**--no-cache-dir功能: 不使用缓存在pip目录下的cache中的文件**

> pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
> 或者
> python setup.py install --cuda_ext --cpp_ext

### 扩展1. `pip install --editable .` vs `python setup.py develop`

[转载自](https://stackoverflow.com/questions/30306099/pip-install-editable-vs-python-setup-py-develop)
Try to avoid calling setup.py directly, it will not properly tell pip that you've installed your package.

**With pip install -e**:

其中 -e 选项全称是--editable. For local projects, the “SomeProject.egg-info” directory is created relative to the project path (**相对于此项目目录的路径**). This is one advantage over just using setup.py develop, which creates the “egg-info” directly relative the current working directory (**相对于当前工作环境目录的路径**).

### 扩展2. 弄懂一个命令 `pip3 install --editable '.[train,test]'` 

[例子在这里](https://github.com/vita-epfl/openpifpaf/blob/21baabf9c6bbd0bea3e8e465a726abfa8dbeeccf/setup.py#L76)
When you then did `pip install --editable .`, the command installs the Python package in the current directory
(signified by the dot .) with the optional dependencies needed for training and
testing ('[train,test]'). 上面的安装命令中，-e选项全称是--editable，也就是可编辑的意思，以可继续开发的模式进行安装，<font color="#dd00dd"> '.' 表示当前目录，也就是setup.py存在的
那个目录，此时pip install将会把包安装在当前文件目录下，而不是安装到所使用的python环境中的-site-packages。</font>
[train,test] 只是我们举的一个例字，是可选参数，在setup.py中可以找到这两个选项（也可能叫其他名字或者根本就没有）之下包含了哪些第三方包。



### 扩展3. 关于egg-info

注意⚠️：选则本地安装`pip install .`成功安装完成后，apex.egg-info文件夹可以只处于当前项目文件夹下而不是安装在系统环境中，只需要在当前使用的python虚拟环境-site-packages中一个指向该egg-info文件的超链接即可(这个是在本地安装自动的行为，不需要我们关心操作)，这样就能找到使用Apex包时所需的apex.egg-info文件夹里的信息。

### PS: 如果遇到Cuda版本不兼容的问题，解决办法见（升级pytoch 1.3后 cuda10.1不匹配版本的警告已经消失）：

[https://github.com/NVIDIA/apex/issues/350\#issuecomment-500390952](https://github.com/NVIDIA/apex/issues/350#issuecomment-500390952)

如果没有出现其他error，可以直接使用上面链接的建议，删除版本检查抛出的报错。见以下讨论：

<https://github.com/NVIDIA/apex/issues/350>

<https://github.com/NVIDIA/apex/pull/323>

### Apex的使用

#### 命令行启动训练

---也是如何Pycharm运行时添加命令行参数的例子

```shell
python -m torch.distributed.launch --nproc_per_node=4 train_distributed.py
```

#### 不使用命令行运行，而是使用Pycharm启动同步夸卡训练的配置

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212185619928.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW9qaWFqaWEwMDc=,size_16,color_FFFFFF,t_70)