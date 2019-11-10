import numpy as np 
from sklearn.model_selection import KFold


def Kfold(num):
    '''
        num: the num of K-Fold
        pathx: the path of x
        pathy: the path of y

        return: a list of train and test

    '''
    x_data = np.loadtxt('./data/exam_x.txt') # data
    y_label = np.loadtxt('./data/exam_y.txt', dtype=int) # label
    KF = KFold(n_splits=num)
    for train, test in KF.split(x_data):
        print(test)

    x = 1


if __name__ == "__main__":
    Kfold(5)
