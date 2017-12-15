## 准备数据

### 1. 下载CIFAR-10数据集

进入kaggle网站，找到[CIFAR-10原始图像分类问题](https://www.kaggle.com/c/cifar-10)，下载提供的数据集。训练和测试集都是7zip的压缩文件，解压需要的时间可能比较久（test数据集我在阿里云上解压了4个多小时才完成）。

7zip文件解压命令为：```7z x [file name] -o[directorname]```

### 2. 初步了解CIFAR-10

CIFAR-10图片格式如下：

+ RGB
+ size：32×32×3
+ 类别：10类，分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车

训练集train.7z：5万张图片，测试集test.7z：30万张图片（Linux可通过命令：ls -lR|grep "^-"|wc -l查看图片数），标签文件为：trainLabels.csv

### 3. 归档存放训练文件

对比之前的FashionMNIST，那种已经被很好的集成了，使用标准的导入函数就可以了，但是现在我们的数据集解压出来后是一堆离散的图片，图片的名称表示其序号，其对应的Label需要在trainLabel.csv文件中找，为了方便查找不同类型的图片，可以有如下方法来处理：

1. 保持图片在一个统一的文件夹中，读取trainLabels.csv文件，将一种类型的图片的序号分别记录在各自的list中，这是时候随机取一个图片需要可以从总的list中查找序号就能知道其类别，要指定取一种类型的图像需要从各自对应类型list中选一个序号然后再总的图片库中找即可

2. 读取trainLabels.csv文件，根据其分类将图片分放的不同的文件夹下，这是时候随机取一个图只需知道它的目录就可以知道其类型，要指定取一种类型的图像只需找到对应的目录即可

从上面分析看起来，两种方法都可以，不过第二种方法需要多一步创建文件夹分类存放文件的操作，也更符合人查看的习惯，比较不会容易出错，我这里也以使用第二方法存放。

读取trainLabels.csv文件并检查：

```{.python .input  n=1}
import pandas as pd

dataLabel = pd.read_csv('trainLabels.csv')
print('shape: ',dataLabel.shape)
dataLabel.head()
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "shape:  (50000, 2)\n"
 },
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>label</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>frog</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>truck</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>truck</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>deer</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>automobile</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   id       label\n0   1        frog\n1   2       truck\n2   3       truck\n3   4        deer\n4   5  automobile"
  },
  "execution_count": 1,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

取得不同的类型和其对应的图片序列号集合：

```{.python .input  n=7}
grouped = dataLabel.groupby('label')
img_id = dataLabel.columns[0]
typedic = {name:list(group[img_id]) for name, group in grouped}
grouped.size()
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "label\nairplane      5000\nautomobile    5000\nbird          5000\ncat           5000\ndeer          5000\ndog           5000\nfrog          5000\nhorse         5000\nship          5000\ntruck         5000\ndtype: int64"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

使用python标准库**shutil**来执行文件、文件夹的操作：

```{.python .input  n=8}
import os
import shutil

def classify_file_by_fold(type_dic, root, dest_root=None):
    if dest_root is None: dest_root = root
    # 创建类别文件夹
    for folder in type_dic.keys():
        path = os.path.join(dest_root,folder)
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 移动图片到对应的文件夹
    for key in type_dic:
        for img in type_dic[key]:
            img = str(img) + '.png'
            src_file = os.path.join(root, img)
            des_file = os.path.join(dest_root, str(key), img)
            if os.path.exists(src_file) and not os.path.exists(des_file):
                shutil.move(src_file, des_file)
    print('File Classify Move Finish')
```

按比例选择一部分图片出来用作测试：

```{.python .input  n=9}
import random

def sel_img_for_test(rate, root, dest_root=None):
    if dest_root is None: 
        foldpath = os.path.join(root,'test')
    else:
        foldpath = dest_root
    # 创建类别验证文件夹
    if not os.path.exists(foldpath):
        os.makedirs(foldpath)

    # 找到所有类别文件夹和文件列表
    dir_list = []
    dir_dic = {}
    return_flag = False
    for root_dir, dirs, files in os.walk(foldpath):
        if root_dir is foldpath and len(files) != 0 or len(dirs) != 0:
            return_flag = True
    for root_dir, dirs, files in os.walk(root):
        if root_dir is root:
            dir_list = [os.path.join(root,dirc) for dirc in dirs]
        elif root_dir == foldpath:
            if len(files) != 0 and len(dirs) != 0:
                return_flag = True
        elif root_dir in dir_list:
            dir_dic[root_dir] = files
            print(root_dir, ',',len(files))
    if return_flag: 
        print('Test files has exist!')
        return
    
    # 随机选取rate比例的文件到测试文件夹
    for src_path in dir_dic:
        files = dir_dic[src_path]
        filenum = int(len(files)*rate)
        files = random.sample(files, filenum)
        for file in files:
            src_file = os.path.join(src_path, file)
            des_file = os.path.join(foldpath, file)
            shutil.move(src_file, des_file)
```

## 数据导入

### 1. 使用Data Augmentation来防止数据过拟合

在深度学习中，当数据量不够大时候，为了防止过拟合，通常有如下方法：

+ Data Augmentation：数据增强，人工增加训练集的大小. 通过平移, 翻转, 加噪声等方法从已有数据中创造出一批"新"的数据
+ Regularization：正则化，通过在Loss Function 后面加上正则项可以抑制过拟合的产生. 缺点是引入了一个需要手动调整的hyper-parameter
+ Dropout：这也是一种正则化手段. 不过跟以上不同的是它通过随机将部分神经元的输出置零来实现

在数据导入时我们可以使用gluon对图像进行数据增强。

```{.python .input  n=10}
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as ag
from mxnet import gluon
from mxnet import image

def data_transform_train(data, label):
    im = data.astype('float32')/255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def data_transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), 
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))
```

### 2. 读取整理后的数据集。

先运行上面的数据整理函数

```{.python .input  n=11}
train_dir = 'train/train/'
train_test_dir = 'train/test/'
test_dir = 'test'
classify_file_by_fold(typedic, train_dir)
sel_img_for_test(0.1, train_dir, train_test_dir)
classify_file_by_fold(typedic, train_test_dir)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "File Classify Move Finish\ntrain/train/dog , 4500\ntrain/train/bird , 4500\ntrain/train/airplane , 4500\ntrain/train/frog , 4500\ntrain/train/automobile , 4500\ntrain/train/cat , 4500\ntrain/train/truck , 4500\ntrain/train/ship , 4500\ntrain/train/deer , 4500\ntrain/train/horse , 4500\nTest files has exist!\nFile Classify Move Finish\n"
 }
]
```

然后使用Gluon中的ImageFolderDataset类来以目录读取图像数据,用DataLoader来做数据加载器

```{.python .input  n=12}
from mxnet.gluon.data import vision

train_ds = vision.ImageFolderDataset(train_dir, flag=1, transform=data_transform_train)
train_test_ds = vision.ImageFolderDataset(train_test_dir, flag=1, transform=data_transform_test)
test_ds = vision.ImageFolderDataset(test_dir, flag=1, transform=data_transform_test)

batch_size = 128
train_data = gluon.data.DataLoader(train_ds, batch_size, shuffle=True, last_batch='keep')
train_test_data = gluon.data.DataLoader(train_test_ds, batch_size, shuffle=True, last_batch='keep')
test_data = gluon.data.DataLoader(test_ds, batch_size, shuffle=False, last_batch='keep')
```

### 3. 查看图片

```{.python .input  n=13}
from matplotlib import pyplot as plt

def show_img(img_datas):
    num = len(img_datas)
    train_smap_datas = [item[0] for item in train_smap]
    train_smap_labels = [item[1] for item in train_smap]
    _, figs = plt.subplots(1, num, figsize=(15,15))
    for i in range(num):
        data = plt.imread(train_smap_datas[i])
        figs[i].imshow(data)
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
    train_label_names = [train_ds.synsets[i] for i in train_smap_labels]
    print(train_label_names)

train_smap = random.sample(train_ds.items, 10)
show_img(train_smap)
```

```{.json .output n=13}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f2a94165c88>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['bird', 'airplane', 'airplane', 'cat', 'truck', 'automobile', 'dog', 'bird', 'bird', 'deer']\n"
 }
]
```

## 定义网络模型

### 1. 检查是否有GPU

```{.python .input  n=14}
try:
    mctx = mx.gpu()
    _ = nd.zeros((1,), ctx=mctx)
except:
    mctx = mx.cpu()
mctx
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 2. 定义需要使用的网络模型

定义模型

```{.python .input  n=15}
from mxnet.gluon import nn

def get_mpl():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(1024*4, activation="relu"),
            nn.Dense(512*4, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(10),
        )
    net.initialize(ctx=mctx)
    return net

def get_lenet():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=50, kernel_size=5,activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=100, kernel_size=3,activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Dense(128),
            nn.Dense(10),
        )
    net.initialize(ctx=mctx)
    return net

def get_alexnet():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
#             # 第一阶段
#             nn.Conv2D(channels=96, kernel_size = 11, strides=2, activation='relu'),
#             nn.MaxPool2D(pool_size=3, strides=2),
            # 第二阶段
            nn.Conv2D(channels=96, kernel_size=5, strides=2, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第三阶段
#             nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=256, kernel_size=3,padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第四阶段
            nn.Flatten(),
            nn.Dense(1024, activation="relu"),
            nn.Dropout(.5),
            # 第五阶段
#             nn.Dense(4096, activation="relu"),
#             nn.Dropout(.5),
            # 第六阶段
            nn.Dense(10)
        )
        net.initialize(ctx=mctx)
        return net
    
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

    
class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 模块1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            # 模块2
            for _ in range(3):
                net.add(Residual(channels=32))
            # 模块3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # 模块4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # 模块5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

from mxnet import init
def get_net(net_name, ctx=mx.cpu()):
    net_name = net_name.upper()
    if net_name == 'LENET':
        return get_lenet()
    elif net_name == 'MPL':
        return get_mpl()
    elif net_name == 'ALEXNET':
        return get_alexnet()
    elif net_name == 'RESNET':
        num_outputs = 10
        net = ResNet(num_outputs)
        net.initialize(ctx=ctx, init=init.Xavier())
        return net
    else:
        print('No Net Named:',net_name)
```

### 3. 定义优化方法对象

```{.python .input  n=16}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练模型

### 1. 定义训练函数

```{.python .input  n=19}
import sys
sys.path.append('../../')
import utils
import time

def train(net, data_iters, lr, wd, epochs, mctx):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'momentum': 0.9, 'wd': wd})
    acc_lists = [] 
    acc_mean = []
    test_acc_list = []
    time_cnt = time.clock()
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        block_cnt = 0
        acc_list = []
        for data_iter in data_iters:
            block_cnt += len(data_iter)
            for data, label in data_iter:
                with ag.record():
                    yhat = net(data.as_in_context(mctx))
                    loss = softmax_cross_entropy(yhat, label.as_in_context(mctx))
                loss.backward()
                trainer.step(len(data))                
                acc = utils.accuracy(yhat, label.as_in_context(mctx))                
#                 print(acc)
                train_loss += nd.mean(loss).asscalar()
                train_acc += acc    
                acc_list.append(acc)
        acc_lists.append(acc_list)
        train_acc_mean = train_acc/block_cnt
        acc_mean.append(train_acc_mean)
        test_acc = utils.evaluate_accuracy_gluon(train_test_data,net,mctx)
        test_acc_list.append(test_acc)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f, Cost Time %f" % (epoch, train_loss/block_cnt, train_acc_mean, test_acc, 
                                                                               time.clock()-time_cnt))
        time_cnt = time.clock()
    return acc_lists, acc_mean, test_acc_list
```

### 2. 选择网络训练

创建网络模型，包括初始化，如果需要保持上次的参数则不再执行下面代码：

```{.python .input  n=20}
# lenet = get_net('LeNet')
# mplnet = get_net('MPL')
# alexnet = get_net('AlexNet')
resnet = get_net('ResNet')
resnet
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "ResNet(\n  (net): HybridSequential(\n    (0): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n    (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    (2): Activation(relu)\n    (3): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (4): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (5): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (6): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv3): Conv2D(None -> 64, kernel_size=(1, 1), stride=(2, 2))\n    )\n    (7): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (8): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (9): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv3): Conv2D(None -> 128, kernel_size=(1, 1), stride=(2, 2))\n    )\n    (10): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (11): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (12): AvgPool2D(size=(8, 8), stride=(8, 8), padding=(0, 0), ceil_mode=False)\n    (13): Flatten\n    (14): Dense(None -> 10, linear)\n  )\n)"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

训练模型

```{.python .input  n=16}
# mplnet params
# learning_rate = 0.003
# weight_decay  = 0.01
# epoch_num    = 5
# net = mplnet

# lenet params
# learning_rate = 0.03
# weight_decay  = 0.001
# epoch_num    = 20
# net = lenet

# alexnet params
# learning_rate = 0.001
# weight_decay  = 0.001
# epoch_num    = 10
# net = alexnet

# resnet params
learning_rate = 0.001
weight_decay  = 0.001
epoch_num    = 50
net = resnet

net.hybridize()
start_time = time.clock()
accs, acc_mean, test_accs = train(net, [train_data], learning_rate, weight_decay, epoch_num, mctx)
acc = []
for i in range(epoch_num):
    acc += accs[i]
plt.plot(acc[::10], '-*r')
plt.show()
print('Total Time:', time.clock()-start_time)
plt.plot(np.array(acc_mean), '-*g')
plt.plot(np.array(test_accs), '-.r')
plt.show()
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 1.696454, Train acc 0.379947, Test acc 0.442383, Cost Time 1884.716236\nEpoch 1. Loss: 1.357515, Train acc 0.506607, Test acc 0.530664, Cost Time 1878.526788\n"
 }
]
```

### 3. 预测未分类的图像

```{.python .input  n=1}
# net.hybridize()
train(net, [train_data,train_test_data], learning_rate, weight_decay, epoch_num, mctx)

preds = []
start_time = time.clock()
for data, label in test_data:
    output = net(data.as_in_context(mctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())
print('Total Time:', time.clock()-start_time)

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_test_ds.synsets[x])
df.to_csv('submission.csv', index=False)
print('Finish')
```

```{.json .output n=1}
[
 {
  "ename": "NameError",
  "evalue": "name 'train' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-1-01c8005b5246>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# net.hybridize()\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mtrain_data\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mtrain_test_data\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight_decay\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepoch_num\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mpreds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mstart_time\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclock\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'train' is not defined"
  ]
 }
]
```

## 总结

+ 使用MPL训练，迭代20次，训练精度可以上升到84%，但是测试精度却只能到达50%，显然过拟合了；增加weight_decay可以稍微提高一点点测试精度，同时训练进度也会降低；降低模型复杂度（如减少每层的节点数），训练精度降低，但是对测试精度没什么改善；加入Dropout层，收敛变得奇慢无比，而且对测试精度没有提高。
+ 使用最简单的卷积神经网络lenet，相比全连接的MPL，对过拟合情况有了很大的改善明显。不过其收敛速度比MPL慢，20次迭代，训练精度为78%（未完全收敛），测试精度智能达到70%；在全部样本训练后分类测试集，第一次提交，精度达到72%。
+ 对于AlexNet，由于机器计算量不够，我不想将图像变大后满足AlexNet的输入要求在做，所以将原来的AlexNet减少几层卷积层做，20次迭代，训练精度为78%（未完全收敛），测试精能达到74%，最后提交经过了40次迭代，精度为0.7769%。回到家，用自己的电脑的GPU训练，速度瞬间提升几倍，顿时感觉GPU简直是神器。
