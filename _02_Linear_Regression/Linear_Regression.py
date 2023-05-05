# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y= read_data()
    #Xw=y
    #(X^T X)^-1 X^T X w = (X^T X)^-1 (X^T y)
    # w = (X^T X)-1 (X^T y)
    weight=np.matmul(np.linalg.inv(np.matmul(X.T, X)),np.matmul(X.T, y))
    return weight @ data
    pass
    
def lasso(data):
    return ridge(data)
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
