import numpy as np 
import matplotlib.pyplot as plt 

def figure(x, y, predict):
    # get teh range of teh picture
    x_min, x_max = x[:, 0].min() -1, x[:, 0].max() + 1 
    y_min, y_max = x[:, 1].min() -1, x[:, 1].max() + 1 
    xaxis, yaxis = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    predict_x = np.c_[xaxis.ravel(), yaxis.ravel()]

    predict_value = predict(predict_x)




    plt.figure()
    # red yellow blue
    plt.contourf(xaxis, yaxis, predict_value, cmap=plt.cm.RdYlBu)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('classfication')
    plt.show()



def fig(x, y):
    # # 画图
    plt.figure()
    plt.scatter(x,y)
    # # plt.plot(x_data,prediction_value,'r-', lw=5)
    plt.show()