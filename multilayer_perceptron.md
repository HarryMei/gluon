## 前言

前面我们实现了线性回归和多类逻辑回归学习模型，不过它们的一个共同点是全是只含有一个输入层，一个输出层，由于它们结构的简单性导致它们在处理负责的问题时得不到很好的效果。

而多层神经网络，就是包含至少一个隐含层的网络，所以这里我使用多层感知机重做softmax回归的fashionMNIST图像数据集的分类器，对比看看精度是否有提升。

## 手动实现多层感知机

前面写过的部分代码放到utils文件中，便于重复调用。

```{.python .input  n=3}
import sys
sys.path.append('./')
import utils

batch_size = 256
train_data, test_data = utils.fashionMNIST_iter(batch_size)
```

相比原来的softmax增加一个隐含层，隐含层的一个节点的输入到输出使用relu作为激活函数，最终的输出仍由softmax来做：

```{.python .input  n=6}
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
import time
import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return nd.maximum(x,0)

def mlp_net(x, ks):
    x = x.reshape((-1,28*28))
    in1 = utils.linear_module(x, ks[0], ks[1])
    out1 = relu(in1)
    out = utils.linear_module(out1, ks[2], ks[3])
    return utils.softmax(out)

def mlp_learning(data_iter, params, iter_num, learning_rate):
    acc_list = []
    loss_list = []
    for epoch in range(iter_num):
        train_loss = 0.
        train_acc = 0.
        for data, label in data_iter:
            with ag.record():
                yhat = mlp_net(data, params)
                loss = utils.cost_fuction_crossentropy(yhat, label)
            loss.backward()
            utils.grad_descent_list(params, learning_rate/(epoch+1))
            
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(yhat, label)
        acc = train_acc/len(data_iter)
        loss_it = train_loss/len(data_iter)
        acc_list.append(acc)
        loss_list.append(loss_it)
        print('Epoch %d. Loss: %f, Train acc %f'%(epoch, loss_it, acc))
    return acc_list, loss_list

img_size = 28*28
hide_node_num = 256
output_node_num = 10
weight_scale = .1
w1 = nd.random_normal(shape=(img_size, hide_node_num), scale=weight_scale)
b1 = nd.zeros((hide_node_num))
w2 = nd.random_normal(shape=(hide_node_num, output_node_num), scale=weight_scale)
b2 = nd.zeros((output_node_num))
params = [w1, b1, w2, b2]
for param in params: param.attach_grad()

print('Before training, Test acc %f:'%(utils.evaluate_accuracy(test_data, mlp_net, params)))

iter_num = 20
learning_rate = 0.001
start_time = time.clock()
acc_list, lost_list = mlp_learning(train_data, params, iter_num, learning_rate)
end_time = time.clock()
print('Training time: %f s \t Each iterator: %f s'% ((end_time-start_time), (end_time-start_time)/iter_num))

plt.plot(np.arange(len(acc_list)),np.array(acc_list), '-*r')
plt.show()
plt.plot(np.arange(len(lost_list)),np.array(lost_list), '-ob')
plt.show()

test_acc = utils.evaluate_accuracy(test_data, mlp_net, params)
print("Test acc %f: " % (test_acc))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before training, Test acc 0.113770:\nEpoch 0. Loss: 174.230584, Train acc 0.769326\nEpoch 1. Loss: 112.059656, Train acc 0.846121\nEpoch 2. Loss: 104.516849, Train acc 0.855912\nEpoch 3. Loss: 100.283466, Train acc 0.862373\nEpoch 4. Loss: 97.698878, Train acc 0.865625\nEpoch 5. Loss: 95.796021, Train acc 0.869099\nEpoch 6. Loss: 94.488339, Train acc 0.869880\nEpoch 7. Loss: 93.224685, Train acc 0.871410\nEpoch 8. Loss: 92.279242, Train acc 0.873077\nEpoch 9. Loss: 91.409163, Train acc 0.874629\nEpoch 10. Loss: 90.710134, Train acc 0.874828\nEpoch 11. Loss: 90.105885, Train acc 0.876463\nEpoch 12. Loss: 89.566167, Train acc 0.876114\nEpoch 13. Loss: 89.055741, Train acc 0.877604\nEpoch 14. Loss: 88.633731, Train acc 0.877704\nEpoch 15. Loss: 88.171601, Train acc 0.878862\nEpoch 16. Loss: 87.812030, Train acc 0.878934\nEpoch 17. Loss: 87.475807, Train acc 0.879383\nEpoch 18. Loss: 87.143264, Train acc 0.879904\nEpoch 19. Loss: 86.830761, Train acc 0.880568\nTraining time: 138.879316 s \t Each iterator: 6.943966 s\n"
 },
 {
  "data": {
   "image/png": "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\nA2F/0QYshse2AAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f92a12c4d68>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f92761e9cc0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Test acc 0.877344: \n"
 }
]
```

**注意：**

*这段程序调试了半天都没有收敛，梯度求出来老是等于0，不知道为什么。后来对比与向导文档中的代码，发现有一点不一样，我一般习惯把线性的系数w和偏置b都合成一个系数矩阵，但是由于mlp是多层结构，结果中间的隐藏层忘记的加偏置节点。*

*我尝试着在net中将隐藏层的输出层加入一个偏置节点，但是却被提示当recording with autograd的时候不支持(+=, -=, x[:]=等切分操作。没有找到好的解决办法，还是老老实实的将线性参数拆成w和b两个。*

*以上的问题全部都修改完后，结果还是不收敛，最后发现我的参数全部都初始化为0了，而向导文档是随机初始化的，于是找文档修改就可以了......*

## 使用gluon提供的模型

```{.python .input  n=7}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy_gluon(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.711145, Train acc 0.736708, Test acc 0.826074\nEpoch 1. Loss: 0.468540, Train acc 0.827460, Test acc 0.850488\nEpoch 2. Loss: 0.410042, Train acc 0.849906, Test acc 0.866797\nEpoch 3. Loss: 0.382016, Train acc 0.860444, Test acc 0.870801\nEpoch 4. Loss: 0.356994, Train acc 0.868390, Test acc 0.869727\nEpoch 5. Loss: 0.341690, Train acc 0.873759, Test acc 0.871777\nEpoch 6. Loss: 0.324127, Train acc 0.880906, Test acc 0.880566\nEpoch 7. Loss: 0.313233, Train acc 0.884680, Test acc 0.886523\nEpoch 8. Loss: 0.304549, Train acc 0.888436, Test acc 0.889160\nEpoch 9. Loss: 0.292557, Train acc 0.890902, Test acc 0.891797\n"
 }
]
```

## 总结

+ 单层的softmax多分类器：0.84 
    + 精度稍低
    + 收敛快
+ 两层多感知机：0.88 
    + 隐含层节点数量越多进度越高，不过过多会出错 
    + 进度比单层的高了一点，但是所用迭代次数和花费的时间都增加
    + 对参数初始值更敏感
