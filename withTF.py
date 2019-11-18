import numpy as np 
import tensorflow as tf 
# import matplotlib.pyplot as plt 
import figure
# import time
import dataprocess

def tfnn(x_data, y_label, batch_size, classnum, hiddennode, itr, learning_rate, activate_fun):


    # figure.fig(x_data[:,0], x_data[:, 1])
    datasize = x_data.shape[0]
    x_data[:, 0] = (x_data[:, 0] - np.mean(x_data[:, 0])) / np.std(x_data[:, 0])
    x_data[:, 1] = (x_data[:, 1] - np.mean(x_data[:, 1])) / np.std(x_data[:, 1])

    # 定义两个placeholder
    x = tf.placeholder(tf.float32,[None, 2])
    y = tf.placeholder(tf.int32,[None, classnum])

    # one_hot_y = tf.one_hot(tf.cast(y, tf.int32), depth=classnum)
    # 定义神经网络中间层 
    Weight_L1 = tf.Variable(tf.random.normal([2, hiddennode]))
    # Weight_L1 = tf.Variable(tf.zeros([2,10]))
    biases_L1 = tf.Variable(tf.zeros([1, hiddennode]))
    Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
    L1 = activate_fun(Wx_plus_b_L1)

    # 定义神经网络输出层
    # Weight_L2 = tf.Variable(tf.random.normal([10,2]))
    Weight_L2 = tf.Variable(tf.zeros([hiddennode, classnum]))
    biases_L2 = tf.Variable(tf.zeros([1,classnum]))
    Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
    prediction = activate_fun(Wx_plus_b_L2)

    # 二次代价函数
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.cast(y, tf.float32)-prediction), axis=1))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,logits=prediction))

    # 使用梯度下降法
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 计算准确率
    # correct_prediction = tf.equal(tf.argmax(one_hot_y, 1),tf.argmax(prediction, 1))
    # accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    batches = dataprocess._get_batches(x_data, y_label, batch_size=batch_size, classnum=classnum)
    losslist = []
    parameters = dict()
    w1list = []
    b1list = []
    w2list = []
    b2list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(itr):
            for batch in batches:
                npx = np.squeeze(batch[0], axis=0)
                npy = np.squeeze(batch[1], axis=0)

                fetch_dict = {
                    'loss': loss,
                    'train': train_step,
                    'predict': prediction,
                    'w1': Weight_L1,
                    'w2': Weight_L2,
                    'b1': biases_L1,
                    'b2': biases_L2
                    # 'accurary': accurary
                }
                
                result = sess.run(fetch_dict, feed_dict={x:npx, y:npy})
            print('itr:{} loss:{}'.format(i, result['loss']))
            w1list.append(result['w1'])
            b1list.append(result['b1'])
            w2list.append(result['w2'])
            b2list.append(result['b2'])
            losslist.append(result['loss'])

        parameters['w1'] = w1list
        parameters['b1'] = b1list
        parameters['w2'] = w2list
        parameters['b2'] = b2list
        parameters['loss'] = losslist


        # # 获取预测值
        # prediction_value = sess.run(prediction, feed_dict={x:x_data})
    return parameters

def predict(x, num, parameters, activate_fun):

    w1, b1 = parameters['w1'][num], parameters['b1'][num]
    w2, b2 = parameters['w2'][num], parameters['b2'][num]

    # Forward
    tfw1 = tf.placeholder(tf.float64, shape=w1.shape, name='w1')
    tfw2 = tf.placeholder(tf.float64, shape=w2.shape, name='w2')
    tfb1 = tf.placeholder(tf.float64, shape=b1.shape, name='b1')
    tfb2 = tf.placeholder(tf.float64, shape=b2.shape, name='b2')
    
    L1 = activate_fun(tf.matmul(x, tfw1) + tfb1)

    L2 = activate_fun(tf.matmul(L1, tfw2) + tfb2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            tfw1: w1,
            tfb1: b1,
            tfw2: w2,
            tfb2: b2,
        }
        # sess.run(L2, feed_dict=feed_dict)
        result = sess.run(tf.argmax(L2, axis=1), feed_dict=feed_dict)
    return result


if __name__ == "__main__":
    
    # x_data = np.loadtxt('./data/exam_x.txt') # data
    # y_label = np.loadtxt('./data/exam_y.txt', dtype=int) # label

    x_data = np.loadtxt('./data/iris_x.txt') # data
    y_label = np.loadtxt('./data/iris_y.txt', dtype=int) # label

    batch_size = 10
    classnum = 2
    hiddennode = 10
    itr = 1000
    learning_rate = 0.05
    activate_fun=tf.nn.sigmoid

    parameters = tfnn(x_data, y_label, batch_size, classnum, hiddennode, itr, learning_rate, activate_fun=activate_fun)
    figure.figurefortf(x_data, y_label, predict=predict, parameters=parameters, acfun=activate_fun)
