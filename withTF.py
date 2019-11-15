import numpy as np 
import tensorflow as tf 
# import matplotlib.pyplot as plt 
import figure
import time

batch_size = 8
classnum = 2

# x_data = np.loadtxt('./data/exam_x.txt') # data
# x_data = np.loadtxt('./data/iris_x.txt') # data
# y_label = np.loadtxt('./data/exam_y.txt', dtype=int) # label
# y_label = np.loadtxt('./data/iris_y.txt', dtype=int) # label
x_data = np.random.random([20000, 2])
y_label = np.zeros([20000,])
for i in range(x_data.shape[0]):
    y_label[i] = int(x_data[i,0] < x_data[i, 1])

# figure.fig(x_data[:,0], x_data[:, 1])
datasize = x_data.shape[0]
# x_data[:, 0] = (x_data[:, 0] - np.mean(x_data[:, 0])) / np.std(x_data[:, 0])
# x_data[:, 1] = (x_data[:, 1] - np.mean(x_data[:, 1])) / np.std(x_data[:, 1])
# x_data[:, 1] /= (x_data[:, 1].max() - x_data[:, 1].min()) 
# figure.fig(x_data[:,0], x_data[:, 1])
# x_data = np.loadtxt('D:\\Python\\machinelearning\\data\\iris_x.txt') # data
# x_list = []
# y_list = []
# print(type(x_data))
# print(x_data.shape)
# x_data[5] = x_data[5][np.newaxis, :]
# x_x = np.expand_dims(x_data[5], axis=0)
# print(x_x.shape)
# exit()
# y_label = np.loadtxt('D:\\Python\\machinelearning\\data\\iris_y.txt') # label
# print(y_label[5])

# y_vector = np.zeros([y_label.shape[0], classnum])
# for i in range(y_label.shape[0]):
#     y_vector[i][y_label[i]] = 1.0
# exit()

# for i in range(x_data.shape[0] // batch_size):
#     x_list.append(x_data[i * batch_size : (i+1) * batch_size, ])
#     y_list.append(y_label[i * batch_size : (i+1) * batch_size, ])
# 定义两个placeholder
x = tf.placeholder(tf.float32,[None, 2])
y = tf.placeholder(tf.float32,[None, 1])

# 定义神经网络中间层
Weight_L1 = tf.Variable(tf.random.normal([2,20]))
# Weight_L1 = tf.Variable(tf.zeros([2,10]))
biases_L1 = tf.Variable(tf.zeros([1,20]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
L1 = tf.nn.sigmoid(Wx_plus_b_L1)

# 定义神经网络输出层
# Weight_L2 = tf.Variable(tf.random.normal([10,2]))
Weight_L2 = tf.Variable(tf.zeros([20,2]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
prediction = tf.nn.sigmoid(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# 使用梯度下降法
# train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        # randid = np.random.randint(0,80)
        start = (i * batch_size) % datasize
        end = min(start+batch_size,datasize)
        # for j in range(batch_size):
        #     batchx = 
        fetch_dict = {
            'loss': loss,
            'train': train_step,
            'predict': prediction,
            'accurary': accurary
        }
        
        # x_test = np.expand_dims(x_data[randid], axis=0)
        # y_test = np.expand_dims(y_label[randid], axis=0)
        result = sess.run(fetch_dict, feed_dict={x:x_data[start:end], y:y_label[start:end][:,np.newaxis]})
        print('itr:{} loss:{}'.format(i, result['loss']))
        # print('Weight_L1:', sess.run(Weight_L1), 'biases_L1:', sess.run(biases_L1))
        # print('itr:{} accurary:{}'.format(i, result['accurary']))
        # time.sleep(0.5)

    # # 获取预测值
    # prediction_value = sess.run(prediction, feed_dict={x:x_data})

