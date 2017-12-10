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

```{.python .input  n=2}
import pandas as pd

dataLabel = pd.read_csv('trainLabels.csv')
print('shape: ',dataLabel.shape)
dataLabel.head()
```

```{.json .output n=2}
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
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

取得不同的类型和其对应的图片序列号集合：

```{.python .input  n=3}
grouped = dataLabel.groupby('label')
img_id = dataLabel.columns[0]
typedic = {name:list(group[img_id]) for name, group in grouped}
grouped.size()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "label\nairplane      5000\nautomobile    5000\nbird          5000\ncat           5000\ndeer          5000\ndog           5000\nfrog          5000\nhorse         5000\nship          5000\ntruck         5000\ndtype: int64"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

使用python标准库**shutil**来执行文件、文件夹的操作：

```{.python .input  n=4}
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

```{.python .input  n=5}
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

```{.python .input  n=2}
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

```{.python .input  n=7}
train_dir = 'train/train/'
train_test_dir = 'train/test/'
test_dir = 'test'
classify_file_by_fold(typedic, train_dir)
sel_img_for_test(0.1, train_dir, train_test_dir)
classify_file_by_fold(typedic, train_test_dir)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "File Classify Move Finish\ntrain/train/dog , 4500\ntrain/train/bird , 4500\ntrain/train/airplane , 4500\ntrain/train/frog , 4500\ntrain/train/automobile , 4500\ntrain/train/cat , 4500\ntrain/train/truck , 4500\ntrain/train/ship , 4500\ntrain/train/deer , 4500\ntrain/train/horse , 4500\nTest files has exist!\nFile Classify Move Finish\n"
 }
]
```

然后使用Gluon中的ImageFolderDataset类来以目录读取图像数据,用DataLoader来做数据加载器

```{.python .input  n=8}
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

```{.python .input  n=9}
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

```{.json .output n=9}
[
 {
  "data": {
   "image/png": "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\n/WtJX1YAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f05017823c8>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['horse', 'airplane', 'bird', 'deer', 'automobile', 'frog', 'frog', 'truck', 'automobile', 'bird']\n"
 }
]
```

## 定义网络模型

### 1. 检查是否有GPU

```{.python .input  n=10}
try:
    mctx = mx.gpu()
    _ = nd.zeros((1,), ctx=mctx)
except:
    mctx = mx.cpu()
mctx
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 2. 定义需要使用的网络模型

定义模型

```{.python .input  n=13}
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

```{.python .input  n=12}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练模型

### 1. 定义训练函数

```{.python .input  n=13}
import sys
sys.path.append('../../')
import utils

def train(net, data_iters, lr, wd, epochs, mctx):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'momentum': 0.9, 'wd': wd})
    acc_lists = [] 
    acc_mean = []
    test_acc_list = []
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
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (epoch, train_loss/block_cnt, train_acc_mean, test_acc))
    return acc_lists, acc_mean, test_acc_list
```

### 2. 选择网络训练

创建网络模型，包括初始化，如果需要保持上次的参数则不再执行下面代码：

```{.python .input  n=15}
# lenet = get_net('LeNet')
# mplnet = get_net('MPL')
# alexnet = get_net('AlexNet')
resnet = get_net('ResNet')
resnet
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "ResNet(\n  (net): HybridSequential(\n    (0): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n    (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    (2): Activation(relu)\n    (3): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (4): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (5): Residual(\n      (conv1): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (6): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv3): Conv2D(None -> 64, kernel_size=(1, 1), stride=(2, 2))\n    )\n    (7): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (8): Residual(\n      (conv1): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (9): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv3): Conv2D(None -> 128, kernel_size=(1, 1), stride=(2, 2))\n    )\n    (10): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (11): Residual(\n      (conv1): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n      (conv2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, in_channels=None)\n    )\n    (12): AvgPool2D(size=(8, 8), stride=(8, 8), padding=(0, 0), ceil_mode=False)\n    (13): Flatten\n    (14): Dense(None -> 10, linear)\n  )\n)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

训练模型

```{.python .input  n=16}
import time

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
epoch_num    = 1
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
  "ename": "NameError",
  "evalue": "name 'train' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-16-2b91b71918b8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     27\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhybridize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0mstart_time\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclock\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 29\u001b[0;31m \u001b[0maccs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0macc_mean\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_accs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mtrain_data\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight_decay\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepoch_num\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     30\u001b[0m \u001b[0macc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mepoch_num\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'train' is not defined"
  ]
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
