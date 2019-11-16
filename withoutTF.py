import numpy as np 
import figure
import dataprocess


class NN():
    def __init__(self, config):
        # self.input_dim = config['input_dim']
        # self.output_dim = config['output_dim']
        # self.learning_rate = config['learning_rate']
        # self.node_num = config['nodes_num'] 
        self.config = config
        self.parameters = dict()
        self.loss = []



    def sigmoid(self, xx):
        return 1.0/(1 + np.exp(-xx))

    def sigmoid_derivation(self, yy):
        return yy * (1 - yy)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def model(self, x_data, y_data):
        count = self.config['itrnum']

        num_examples = len(x_data)

        # init paras
        # w1 = np.zeros([self.config['input_dim'], self.config['nodes_num']])
        # w2 = np.zeros([self.config['nodes_num'], self.config['output_dim']])
        w1 = np.random.rand(self.config['input_dim'], self.config['nodes_num'])
        w2 = np.random.rand(self.config['nodes_num'], self.config['output_dim'])
        b1 = np.zeros([1, self.config['nodes_num']])
        b2 = np.zeros([1, self.config['output_dim']])

        w1list = []
        b1list = []
        w2list = []
        b2list = []

        w1list.append(w1)
        b1list.append(b1)
        w2list.append(w2)
        b2list.append(b2)

        # x_data = self._normalization(x_data)
        x_data = dataprocess._normalization_normal(x_data)

        batches = dataprocess._get_batches(x_data, y_data, self.config['batch_size'], self.config['output_dim'])


        for i in range(count):
            for batch in batches:
                x = np.squeeze(batch[0], axis=0)
                y = np.squeeze(batch[1], axis=0)


                # Forward
                w1b1 = np.dot(x, w1) + b1
                L1 = self.sigmoid(w1b1)

                w2b2 = np.dot(L1, w2) + b2
                L2 = self.sigmoid(w2b2)


                # Backpropagation
                error = L2 - y
                erroutput = np.multiply(error, self.sigmoid_derivation(L2))
                errhidden = np.multiply(np.dot(erroutput, w2.T), self.sigmoid_derivation(L1))
                # delta3[range(num_examples), y] -= 1
                # dW2 = (a1.T).dot(delta3)
                # db2 = np.sum(delta3, axis=0, keepdims=True)
                # delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
                # dW1 = np.dot(X.T, delta2)
                # db1 = np.sum(delta2, axis=0)
                w2 = w2 - self.config['learning_rate'] * np.dot(L1.T, erroutput)
                b2 = b2 - self.config['learning_rate'] * erroutput
                w1 = w1 - self.config['learning_rate'] * np.dot(x.T, errhidden)
                b1 = b1 - self.config['learning_rate'] * errhidden

                w1list.append(w1)
                b1list.append(b1)
                w2list.append(w2)
                b2list.append(b2)

            loss = 0.5 * np.mean(np.sum((L2 - y)**2, axis=1))

            if i % 10 == 0:
                print('itr: %i, loss: %.8f' % (i, loss))
            self.loss.append(loss)

        self.parameters['w1'] = w1list
        self.parameters['b1'] = b1list
        self.parameters['w2'] = w2list
        self.parameters['b2'] = b2list

        return self.loss


    def predict(self, x, num):

        w1, b1 = self.parameters['w1'][num], self.parameters['b1'][num]
        w2, b2 = self.parameters['w2'][num], self.parameters['b2'][num]

        # Forward
        L1 = np.dot(x, w1) + b1[0]
        L1 = self.sigmoid(L1)

        L2 = np.dot(L1, w2) + b2[0]
        L2 = self.sigmoid(L2)

        return np.argmax(L2, axis=1)



if __name__ == "__main__":

    x_data = np.loadtxt('./data/exam_x.txt') # data
    y_label = np.loadtxt('./data/exam_y.txt', dtype=int) # label
    # x_data = np.loadtxt('./data/iris_x.txt') # data
    # y_label = np.loadtxt('./data/iris_y.txt', dtype=int) # label
    
    config = dict()
    config['input_dim'] = 2
    config['output_dim'] = 2
    config['learning_rate'] = 0.01
    config['nodes_num'] = 10
    config['batch_size'] = 10
    config['itrnum'] = 2000


    NNmodel = NN(config)

    loss = NNmodel.model(x_data, y_label)
    # figure.lossfig(loss)
    figure.figure(x_data, y_label, predict=NNmodel.predict, loss=loss)




