import tensorflow as tf
import numpy as np

batch_size = 128
test_size = 256

batch_file_1 = "data_batch_1"
batch_file_2 = "data_batch_2"
batch_file_3 = "data_batch_3"
batch_file_4 = "data_batch_4"
batch_file_5 = "data_batch_5"
test_file = "test_batch"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot(labels):
    encoded = np.zeros((len(labels),10))
    for i in range(len(labels)):
        encoded[i][labels[i]] = 1
    return encoded

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

batch_1 = unpickle(batch_file_1)
batch_2 = unpickle(batch_file_2)
batch_3 = unpickle(batch_file_3)
batch_4 = unpickle(batch_file_4)
batch_5 = unpickle(batch_file_5)
data = np.concatenate((batch_1[b'data'],batch_2[b'data'],batch_3[b'data'],
                       batch_4[b'data'],batch_5[b'data']), axis=0)
trY = one_hot(np.concatenate((batch_1[b'labels'],batch_2[b'labels'],batch_3[b'labels'],
                         batch_4[b'labels'],batch_5[b'labels']), axis=0))

trX = data.reshape(-1,32,32,3)
trX = trX/255

test_batch = unpickle(test_file)
test_data = test_batch[b'data']
teX = test_data.reshape(-1,32,32,3)
teY = one_hot(test_batch[b'labels'])

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 3, 32])       # 3x3x1 conv, 32 outputs
w_fc = init_weights([32 * 16 * 16, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print("iteration: ", i, "Accuracy: ", np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))