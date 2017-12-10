# take a short review of numpy and check the two type ndarray difference between numpy and mxnet 

import numpy as np
import mxnet as mn
import mxnet.ndarray as nd

def numpy_test():
    x= np.zeros((3,3))
    print(x)
    y= 2*np.ones((3,3))
    print(y)
    z= np.arange(0,9,1).reshape((3,3))
    print(z)
    print(z*y)
    print(np.exp(x))
    mat_mul = np.dot(z,y)
    print(mat_mul)
    print(z.T)
    
def numpy_array():
    a = np.arange(10)
    b = a
    c = np.copy(a)
    b[0] = 10
    c[2] = 10
    print(a, b, c)
    
def numpy_mat():
    a = np.arange(9).reshape((3,3))
    b = np.array([1,2,3])
    print(a, b)
    print(a*b)
    print(a[:,1]+10)
    c = np.zeros(3)
#     c[:,1] += 2
    print(len(c.shape))
    d = np.eye(3)
    print(np.sum(a))
    
def array_change(arr):
#     arr[:] += 1
    #  numpy在下面代码段中的是值传递，mxnet是引用传递
    for num in arr:
        num += 2
def array_pass():
    arr = nd.ones(2);
    print(arr)
    array_change(arr)
    print(arr)

if __name__ == '__main__':
    print('numpy version', np.__version__)
    print('mxnet version: ', mn.__version__)
    array_pass()
#     numpy_mat()
#     numpy_array()
#     numpy_test()
