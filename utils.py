from mxnet import gluon
from mxnet import autograd as ag
from mxnet import ndarray as nd
from matplotlib import pyplot as plt

######## fashionMNIST数据集导入

def fashionMNIST_load(path):
    def transform(data, label):
        return data.astype('float32')/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root = path, train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(root = path, train=False, transform=transform)
    return mnist_train, mnist_test

def fashionMNIST_iter(batch_size, root = '~/.mxnet/datasets/fashion-mnist'):
    mnist_train, mnist_test = fashionMNIST_load(root)
    train_data = gluon.data.DataLoader(mnist_train, batch_size,shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return train_data, test_data

def show_img(img_data, num):
    data, label = img_data[:num]
    _, figs = plt.subplots(1, num, figsize=(15,15))
    for i in range(num):
        figs[i].imshow(data[i].reshape((28,28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
    print(label)
    
    
######### 模型

def softmax(x):
    exp = nd.exp(x)
    partition = exp.sum(axis=1, keepdims=True)
    return exp/partition

def linear_module(x, w, b):
    return nd.dot(x, w) + b


######### 梯度下降

def grad_descent(params, rate):
    params[:] -= rate*params.grad

def grad_descent_list(params, rate):
    for param in params:
        param[:] -= rate*param.grad

######### 代价函数

def cost_fuction_crossentropy(yhat, y):
    return - nd.sum(nd.log(nd.pick(yhat, y)), axis=0)


######### 精度计算
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, params):
    acc = 0.
    for data, label in data_iterator:
        output = net(data, params)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

import mxnet as mx
def evaluate_accuracy_gluon(data_iterator, net, mctx=mx.cpu()):
    acc = 0.
    for data, label in data_iterator:
        output = net(data.as_in_context(mctx))
        acc += accuracy(output, label.as_in_context(mctx))
    return acc / len(data_iterator)

def evaluate_accuracy_gluon_cnn(data_iterator, net, mctx=mx.cpu()):
    acc = 0.
    for data, label in data_iterator:
        output = net(data.as_in_context(mctx).reshape((-1,1,data.shape[1],data.shape[2])))
        acc += accuracy(output, label.as_in_context(mctx))
    return acc / len(data_iterator)