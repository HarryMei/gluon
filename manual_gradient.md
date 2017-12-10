## 前言

在机器学习算法中，我们总是能在各种地方看到梯度下降（上升）算法，可见梯度下降算法对计算这些机器学习问题最优解的重要性。基本梯度下降算法的思想是非常简单朴素的，它在我们一开始接触机器学习时就能有初步认识，不过也许正是因为简单所以比较容易被忽视，如果真要从头开始将理论转化为梯度下降完整的代码也没那么容易，所以我再次回顾总结加深理解。

## 导数与梯度

所谓梯度下降，就是沿着梯度的方向前进，由于对于凸函数而言沿着梯度减小的方向走函数值是递减的，所以当梯度不能减小到0时便表示已经走到了谷底。那么梯度是什么东西呢？直观的讲梯度就是变化率，以前我就是这吗理解的，不过这里显然漏掉梯度另外一个非常重要的意义：变化最大的方向。所以我们知道了梯度是一个向量，既有大小也有方向。

梯度怎么求？因为导数就是表示函数变化率的，所以很显然通过求导可以得到梯度值。对于单变量函数，导数值就是梯度值；对于多变量函数，通过求各个变量的偏导数得到梯度向量。

## 手动求导

对于那些解析式已知的方程，一般可以通过数学上的求导公式得到的。然而这样求导是针对每一个特定的表达式求得一个导数表达式，当解析式变化时我们需要重新计算其导数表达式，当用计算机来工作的时候，我们更希望能有一个通用的方法求导。这里我们可以通过微分思想，自变量在某一点的导数，取自变量极小的变化下函数值的变化率来近似求得。

当前各个深度学习框架，一个基础的功能就是提供自动求导，可见求导的重要性。不过尽管如此，在使用这些框架提供的自动求导功能前，先手动写一个求导程序会更好。下面便是通过微分求导的python代码。

```{.python .input  n=21}
def get_derivatives(func, point, distance=0.0001):
    left_y = func(point)
    right_y = func(point+distance)
    return (right_y-left_y)/distance
```

测试get_derivatives函数：

```{.python .input  n=25}
def test_expression(x):
    return 2*x*x + 10*x;

point = -20.56
der = get_derivatives(test_expression, point)
print('get_derivatives: ', der)
print('real val: ', 4*point+10)
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "get_derivatives:  -72.23979999821495\nreal val:  -72.24\n"
 }
]
```

## 手动求梯度

对于梯度，只不过是求各个自变量的偏导，但是要注意的是传入的求梯度的位置和梯度的求取结果都是向量，整个方程的表达是也是向量形式的，所以我这里使用numpy的array作为变量的类型：

```{.python .input  n=43}
import numpy as np

def get_gradient(func, point_mat, distance=0.0001):
    gradient = np.zeros(point_mat.shape)
    col = 0
    #  对每一个自变量求偏导，numpy可以对每一列整体操作
    for point in point_mat[0]:
        new_mat = np.copy(point_mat)
        new_mat[:,col] += distance; 
        left_y = func(point_mat)
        right_y = func(new_mat)
        gradient[:,col] = (right_y-left_y)/distance      
        col += 1
    return gradient
```

测试get_gradient函数：

```{.python .input  n=62}
# 表达式：y = k1*x1^2 + k2*x2^2 + ... + kn*xn^2
# 矩阵形式: Y = A.T X.T X
def test_vec_expression(x):
    A = np.arange(x.shape[0])+1
    X = x*x
    return np.dot(A,X.T);

point_mat = np.arange(1,10).reshape((3,3)).astype(np.float)
print('X = \n', point_mat)
print('Y = ',test_vec_expression(point_mat))
print('get_gradient: \n',get_gradient(test_vec_expression,point_mat))
A = np.arange(point_mat.shape[0])+1
print('real val: \n', 2*A*point_mat)
```

```{.json .output n=62}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "X = \n [[ 1.  2.  3.]\n [ 4.  5.  6.]\n [ 7.  8.  9.]]\nY =  [  36.  174.  420.]\nget_gradient: \n [[  2.0001   8.0002  18.0003]\n [  8.0001  20.0002  36.0003]\n [ 14.0001  32.0002  54.0003]]\nreal val: \n [[  2.   8.  18.]\n [  8.  20.  36.]\n [ 14.  32.  54.]]\n"
 }
]
```

## 使用深度学习框架库自动求导

这里我使用**MXNet**这个深度学习框架，所以需要使用到MXNet的基本数据类型和自动求导功能,同样定义和test_vec_expression中一样的表达式，只不过使用MXNet中的数据类型：

```{.python .input  n=84}
import mxnet.ndarray as nd
import mxnet.autograd as ag

def test_autograd_expression(x):
    A = nd.arange(x.shape[0])+1
    X = x*x
    return nd.dot(A,X.T);
```

测试自动求导：

```{.python .input  n=86}
point_mat = nd.arange(1,10).reshape((3,3))
print('X = ', point_mat)
print('Y = ',test_autograd_expression(point_mat))

point_mat.attach_grad()
with ag.record():
    y = test_autograd_expression(point_mat)

y.backward()
print('point_mat.gradient: ', point_mat.grad)
```

```{.json .output n=86}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "X =  \n[[ 1.  2.  3.]\n [ 4.  5.  6.]\n [ 7.  8.  9.]]\n<NDArray 3x3 @cpu(0)>\nY =  \n[  36.  174.  420.]\n<NDArray 3 @cpu(0)>\npoint_mat.gradient:  \n[[  2.   8.  18.]\n [  8.  20.  36.]\n [ 14.  32.  54.]]\n<NDArray 3x3 @cpu(0)>\n"
 }
]
```
