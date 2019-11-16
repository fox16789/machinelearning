import numpy as np 



def Kfold(x, y, num):
    '''
        num: the num of K-Fold
        pathx: the path of x
        pathy: the path of y

        return: a list of train and test

    '''
    x_data = x # data
    y_label = y # label
    assert type(x_data) == type(np.zeros([1,1]))
    alldata = np.concatenate([x, y], axis=1)
    size = alldata.shape[0] // num
    train = []
    test = []
    
    for i in range(num):
        
