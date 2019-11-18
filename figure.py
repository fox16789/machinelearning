import numpy as np 
import matplotlib.pyplot as plt 
import time

def figure(x, y, predict, loss):
    # get teh range of teh picture
    x_min, x_max = x[:, 0].min() -0.1, x[:, 0].max() + 0.1 
    y_min, y_max = x[:, 1].min() -0.1, x[:, 1].max() + 0.1 
    xaxis, yaxis = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    predict_x = np.c_[xaxis.ravel(), yaxis.ravel()]

    plt.ion()
    plt.figure()

    for i in range(0, len(loss), 10):
        # red yellow blue
        plt.subplot(121)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('classfication')
        predict_value = predict(predict_x, i)
        predict_value = np.reshape(predict_value, xaxis.shape)
        plt.contourf(xaxis, yaxis, predict_value, cmap=plt.cm.RdYlBu)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)

        plt.subplot(122)
        plt.title('loss')
        plt.xlabel('itr')
        plt.ylabel('loss')
        plt.plot(i, loss[i], c='r', ls='-', marker='.')
        # plt.draw()
        plt.pause(0.1)
        # time.sleep(0.01)
    plt.pause(30)
    plt.close()




def figurefortf(x, y, predict, parameters, acfun):
    loss = parameters['loss']
    # get teh range of teh picture
    x_min, x_max = x[:, 0].min() -0.1, x[:, 0].max() + 0.1 
    y_min, y_max = x[:, 1].min() -0.1, x[:, 1].max() + 0.1 
    xaxis, yaxis = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    predict_x = np.c_[xaxis.ravel(), yaxis.ravel()]

    plt.ion()
    plt.figure()

    for i in range(0, len(loss), 10):
        # red yellow blue
        plt.subplot(121)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('classfication')
        predict_value = predict(predict_x, i, parameters=parameters, activate_fun=acfun)
        predict_value = np.reshape(predict_value, xaxis.shape)
        plt.contourf(xaxis, yaxis, predict_value, cmap=plt.cm.RdYlBu)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu, marker=(9,3,10))

        plt.subplot(122)
        plt.title('loss')
        plt.xlabel('itr')
        plt.ylabel('loss')
        plt.plot(i, loss[i], c='r', ls='-', marker='.')
        # plt.draw()
        plt.pause(0.01)
        # time.sleep(0.01)
    plt.pause(30)
    plt.close()