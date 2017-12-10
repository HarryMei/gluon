## 前言

CNN（Convolutional Neural Network）卷积神经网络，可谓是当前深度学习在计算机视觉邻域应用的最成功的的技术，本文就不详细说明卷积神经网络的各种原理，反正我也只是只知其一不知其二，因为这里主要是做代码练习的，所以这里我会重点关注CNN的大体实现和使用。

这里主要实现，比较传统的简单的卷积神经网络。

## CNN组成

### 1. 全连接层

前面我们做softmax单层分类器以及多层感知机，他们都是全连接网络。什么事全连接网络？我的理解就是将前层网络中的全部节点都与后面一层网络的全部节点互联。就像我们所做练习那样，图像中每个像素点都会一维向量化，作为独立的节点连接到后一层，这样当图像足够大的时候，网络结构就太复杂了。所以CNN会将图像的一块小区域作为一个整体连接到后面。

### 2. 卷积模块

当然CNN之所以是卷积神经网络肯定是会包括卷积操作的，正如前面说将的那样，CNN不是全部层都是全连接，前面的某一些层是使用一小块区域作为一个连接节点的，而这一小块区域的操作就是卷积。实际上每一个卷积模块都是包括两个部分：一个是卷积操作，一个是池化操作。通常是“卷积层-激活层-池化层”这样组成。

这里就解释什么事卷积什么是池化了，详细可以参考斯坦福大学网页教程：http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B 。

### 3. 组成

实际上传统的CNN就是由几层卷积模块，加上全连接网络组成。

## CNN手动实现

初始化参数,比较值得注意的是权重的shape，卷积模块的权重和全连接网络的权重是不一样的，它需要是四维向量：output_channels x in_channels x height x width，而当网络层越深权重的的shape也需要自己根据网络的结构推算后才能知道（卷积操作会让图像变小，卷积和pooling的stride参数也会让图像变小，权重需要根据输入图像确定），特别是卷积层到全连接层的权重。

另外卷积神经网络会更慢，最好到GPU上来训练。

```{.python .input  n=2}
import sys
sys.path.append('./')
import utils

batch_size = 256
train_data, test_data = utils.fashionMNIST_iter(batch_size)

from mxnet import gluon
import mxnet.ndarray as nd
import mxnet.autograd as ag
import numpy as np
import mxnet as mx

ctx_list = (mx.cpu(), mx.gpu())
mctx = ctx_list[0]

scal = 0.1
out_ch1 = 20
kernel_size1 = [5,5]
w1 = nd.random_normal(shape=(out_ch1,1,kernel_size1[0],kernel_size1[1]), scale=scal, ctx=mctx)
b1 = nd.zeros((out_ch1,),ctx=mctx)

out_ch2 = 50
kernel_size2 = [3,3]
w2 = nd.random_normal(shape=(out_ch2,out_ch1,kernel_size2[0],kernel_size2[1]), scale=scal, ctx=mctx)
b2 = nd.zeros((out_ch2,),ctx=mctx)

hide_node = 128
w3 = nd.random_normal(shape=(1250,hide_node), scale=scal, ctx=mctx)
b3 = nd.zeros((1,),ctx=mctx)

out_node = 10
w4 = nd.random_normal(shape=(hide_node,out_node), scale=scal, ctx=mctx)
b4 = nd.zeros((1,),ctx=mctx)

params = [w1,b1,w2,b2,w3,b3,w4,b4]
for param in params:
    param.attach_grad()
```

使用mxnet提供的功能函数，手动构建一个包含2层卷积模块、2层全连接的网络。输入数据时也需要注意shape，它的格式为：batch x channel x height x width。

在使用nd.Convolution函数做卷积操作时，需要注意的是：**bias的第一个维度数值一定要和weight的第一个维度的数值相等**。

```{.python .input  n=5}
import matplotlib.pyplot as plt
import time

def cnn_net(x, params):
    w1,b1,w2,b2,w3,b3,w4,b4 = params
    #  lay1: convolution
    x = x.reshape((-1,1,x.shape[1],x.shape[2]))
    h1_conv = nd.Convolution(data=x, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2,2), stride=(2,2))
    #  lay2: convolution
    h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    #  lay3: full connected
    h3_linear = nd.dot(h2, w3) + b3
    h3 = nd.relu(h3_linear)
    #  lay4: full connected
    h4_linear = nd.dot(h3, w4) + b4
    
#     print('1st conv block:', h1.shape)
#     print('2nd conv block:', h2.shape)
#     print('1st dense:', h3.shape)
#     print('2nd dense:', h4_linear.shape)
#     print('output:', h4_linear.shape)
    return nd.softmax(h4_linear)
#     return h4_linear

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()   
    
def cnn_learning(data_iter, params, iter_num, learning_rate):
    acc_list = []
    loss_list = []
    for epoch in range(iter_num):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            label = label.as_in_context(mctx)
            data = data.as_in_context(mctx)
            with ag.record():
                yhat = cnn_net(data, params)
                loss = utils.cost_fuction_crossentropy(yhat, label)
#                 loss = softmax_cross_entropy(yhat, label)
            loss.backward()
            utils.grad_descent_list(params, learning_rate/len(label))
#             print(loss, params[1].grad)
            
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(yhat, label)
        acc = train_acc/len(data_iter)
        loss_it = train_loss/len(data)
            
        acc_list.append(acc)
        loss_list.append(loss_it)
        test_acc = utils.evaluate_accuracy(test_data, cnn_net, params)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
    return acc_list, loss_list
        

print('Before training, Test acc %f:'%(utils.evaluate_accuracy(test_data, cnn_net, params)))

iter_num = 5
learning_rate = 0.3
start_time = time.clock()
acc_list, lost_list = cnn_learning(train_data, params, iter_num, learning_rate)
end_time = time.clock()
print('Training time: %f s \t Each iterator: %f s'% ((end_time-start_time), (end_time-start_time)/iter_num))

plt.plot(np.arange(len(acc_list)),np.array(acc_list), '-*r')
plt.show()
plt.plot(np.arange(len(lost_list)),np.array(lost_list), '-ob')
plt.show()

test_acc = utils.evaluate_accuracy(test_data, cnn_net, params)
print("Test acc %f: " % (test_acc))
        
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before training, Test acc 0.109082:\nEpoch 0. Loss: 194.112330, Train acc 0.717354, Test acc 0.789746\nEpoch 1. Loss: 110.344156, Train acc 0.839312, Test acc 0.828125\nEpoch 2. Loss: 93.974803, Train acc 0.862871, Test acc 0.782910\nEpoch 3. Loss: 84.751337, Train acc 0.875720, Test acc 0.883398\nEpoch 4. Loss: 79.209315, Train acc 0.883782, Test acc 0.853516\nTraining time: 210.617324 s \t Each iterator: 42.123465 s\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7ff2943a0208>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7ff26cee4ba8>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Test acc 0.853516: \n"
 }
]
```

## 使用gluon的模型

从上面手动写的CNN部分可以看到，我们需要得预先设置好参数才行，总结手写时遇到的两个坑：

1. 第一个是参数矩阵的shape的确定，需要自己仔细推算，稍不注意就会弄错
2. 参数的初值没有选好，导致结果不收敛，特别是生成随机参数时random_normal中的scale参数，过大过小都会导致导致求导为0

所以使用gluon框架的优势就体现出来了，第一它自动确定各个参数矩阵的shape，第二它自己做初始化，第三它添加、删减层简单，几乎不影响其他代码。

不过反过来说，作为初学者也正式由于这样的手写练习，才能发现或这或那的问题，没有切身体会过就肯定不会留意这些。

```{.python .input  n=6}
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20,kernel_size=5,activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50,kernel_size=3,activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128),
        nn.Dense(10)
    )
net.initialize(ctx=mctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate': 0.3})

def gluon_cnn_learning(data_iter,iter_num):
    acc_list = []
    loss_list = []
    for epoch in range(iter_num):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            label = label.as_in_context(mctx)
            data = data.as_in_context(mctx)
            with ag.record():
                yhat = net(data.reshape((-1,1,data.shape[1],data.shape[2])))
                loss = softmax_cross_entropy(yhat, label)
            loss.backward()
            trainer.step(len(label))
            
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(yhat, label)
        acc = train_acc/len(data_iter)
        loss_it = train_loss/len(data)
            
        acc_list.append(acc)
        loss_list.append(loss_it)
        test_acc = utils.evaluate_accuracy_gluon_cnn(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
    return acc_list, loss_list
        

print('Before training, Test acc %f:'%(utils.evaluate_accuracy_gluon_cnn(test_data, net)))

iter_num = 10
start_time = time.clock()
acc_list, lost_list = gluon_cnn_learning(train_data, iter_num)
end_time = time.clock()
print('Training time: %f s \t Each iterator: %f s'% ((end_time-start_time), (end_time-start_time)/iter_num))

plt.plot(np.arange(len(acc_list)),np.array(acc_list), '-*r')
plt.show()
plt.plot(np.arange(len(lost_list)),np.array(lost_list), '-ob')
plt.show()

test_acc = utils.evaluate_accuracy_gluon_cnn(test_data, net)
print("Test acc %f: " % (test_acc))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before training, Test acc 0.097070:\nEpoch 0. Loss: 1.144225, Train acc 0.591617, Test acc 0.788574\nEpoch 1. Loss: 0.535565, Train acc 0.796709, Test acc 0.823828\nEpoch 2. Loss: 0.456594, Train acc 0.829599, Test acc 0.853125\nEpoch 3. Loss: 0.410379, Train acc 0.849352, Test acc 0.858008\nEpoch 4. Loss: 0.371355, Train acc 0.864888, Test acc 0.859961\nEpoch 5. Loss: 0.346662, Train acc 0.875316, Test acc 0.830469\nEpoch 6. Loss: 0.329518, Train acc 0.880873, Test acc 0.871680\nEpoch 7. Loss: 0.311273, Train acc 0.887677, Test acc 0.833594\nEpoch 8. Loss: 0.302939, Train acc 0.890475, Test acc 0.874512\nEpoch 9. Loss: 0.292357, Train acc 0.893523, Test acc 0.870020\nTraining time: 406.057311 s \t Each iterator: 40.605731 s\n"
 },
 {
  "data": {
   "image/png": "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\nQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7ff278315668>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\nYwkAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7ff26d0bcfd0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Test acc 0.870020: \n"
 }
]
```
