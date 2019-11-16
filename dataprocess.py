import numpy as np 


def _one_hot(y, num):

    sample_num = y.shape[0]
    one_hot = np.zeros((sample_num, num)).astype('int64')
    one_hot[np.arange(sample_num).astype('int64'), y.astype('int64').T] = 1
    return one_hot

def _normalization(x_data):
    x_data[:, 0] = (x_data[:, 0] - x_data[:, 0].min()) / (x_data[:, 0].max() - x_data[:, 0].min()) 
    x_data[:, 1] = (x_data[:, 1] - x_data[:, 1].min()) / (x_data[:, 1].max() - x_data[:, 1].min()) 

    return x_data

def _normalization_normal(x_data):
    x_data[:, 0] = (x_data[:, 0] - np.mean(x_data[:, 0])) / np.std(x_data[:, 0])
    x_data[:, 1] = (x_data[:, 1] - np.mean(x_data[:, 1])) / np.std(x_data[:, 1])

    return x_data

def _get_batches(x, y, batch_size, classnum):

    y = _one_hot(y, classnum)

    batches = []
    for i in range(np.int(x.shape[0] / batch_size)):
        x_batch = []
        y_batch = []
        x_batch.append(x[i * batch_size : (i+1) * batch_size])
        y_batch.append(y[i * batch_size : (i+1) * batch_size])
        batch = (np.array(x_batch), np.array(y_batch))
        batches.append(batch)

    return batches
