import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import time

class NN():
    def __init__(self, classnum, learing_rate, traindata, labeldata):
        
        # batch_size = 8
        self.classnum = classnum

        self.x_data = traindata
        self.y_label = labeldata

        self.learning_rate = learing_rate

        self.bulid_graph()
        self.sess = tf.Session(graph=self.graph)


        # 使用numpy生成200个随机点
        # x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
        # noise = np.random.normal(0,0.02,x_data.shape)
        # y_data = np.square(x_data) + noise
   
    # x_data = np.loadtxt('D:\\Python\\machinelearning\\data\\iris_x.txt') # data
    def bulid_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32,[None, 2])
            self.y = tf.placeholder(tf.float32,[None, 1])
            
            # hidden layer
            w1 = self._weight_variable([2, 5])
            b1 = self._bias_variable([1, 5])
            Wx_plus_b_L1 = tf.matmul(self.x, w1) + b1
            L1 = tf.nn.sigmoid(Wx_plus_b_L1)
            
            # output layer
            w2 = self._weight_variable([5, 2])
            b2 = self._bias_variable([1, 2])
            Wx_plus_b_L2 = tf.matmul(L1,w2) + b2
            self.prediction = tf.nn.softmax(Wx_plus_b_L2)

            # loss
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.prediction)

            # train
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # accurary
            correct_prediction = tf.equal(tf.argmax(self.y, 1),tf.argmax(self.prediction, 1))
            self.accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _weight_variable(self, shape):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        return var

    def _bias_variable(self, shape):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        return var

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for i in range(2000):
            randid = np.random.randint(0,80)
            # for j in range(batch_size):
            #     batchx = 
            fetch_dict = {
                'train': self.train_step,
                'loss': self.loss,
                'predict': self.prediction
            }
            
            x_test = np.expand_dims(x_data[randid], axis=0)
            # y_test = np.expand_dims(y_label[randid], axis=0)
            result = self.sess.run(fetch_dict, feed_dict={self.x:x_test, self.y:y_vector[randid][np.newaxis,:]})
            print('loss:{}'.format(result['loss']))
            print('predict:{}'.format(result['predict']))


    x_list = []
    y_list = []
    # print(type(x_data))
    # print(x_data.shape)
    # x_data[5] = x_data[5][np.newaxis, :]
    x_x = np.expand_dims(x_data[5], axis=0)
    print(x_x.shape)
    # exit()
   
    # y_label = np.loadtxt('D:\\Python\\machinelearning\\data\\iris_y.txt') # label
    print(y_label[5])
    y_vector = np.zeros([y_label.shape[0], classnum])
    for i in range(y_label.shape[0]):
        y_vector[i][y_label[i]] = 1.0
    # exit()


    # # 获取预测值
    # prediction_value = sess.run(prediction, feed_dict={x:x_data})

    # # 画图
    # plt.figure()
    # plt.scatter(x_data[:, 0],x_data[:, 1])
    # # plt.plot(x_data,prediction_value,'r-', lw=5)
    # plt.show()


if __name__ == "__main__":
    x_data = np.loadtxt('./data/exam_x.txt') # data
    y_label = np.loadtxt('./data/exam_y.txt', dtype=int) # label

    model = NN(classnum=2, learing_rate=0.01, traindata=x_data, labeldata=y_label)
    model.train()
    