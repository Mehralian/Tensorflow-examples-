import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
'''
# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
'''

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


n_input=784
n_hidden_1=256
n_hidden_2=128

X = tf.placeholder('float',[None,784])

init_weight = tf.random_normal_initializer()
init_constant = tf.constant_initializer(0.0)


weights = {
    
    'encoder_w1': tf.get_variable('ew1',shape= [n_input,n_hidden_1],initializer=init_weight),
    'encoder_w2': tf.get_variable('ew2',shape=[n_hidden_1,n_hidden_2],initializer=init_weight),
    'decoder_w1': tf.get_variable('dw1',shape= [n_hidden_2,n_hidden_1],initializer=init_weight),
    'decoder_w2': tf.get_variable('dw2',shape=[n_hidden_1,n_input],initializer=init_weight)
}

bias = {
    
    'encoder_b1' : tf.get_variable('eb1',shape=[n_hidden_1],initializer=init_constant),
    'encoder_b2' : tf.get_variable('eb2',shape=[n_hidden_2],initializer=init_constant),
    'decoder_b1' : tf.get_variable('db1',shape=[n_hidden_1],initializer=init_constant),
    'decoder_b2' : tf.get_variable('db2',shape=[n_input],initializer=init_constant)
}



# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
'''
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

'''

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()


    '''
Epoch: 0001 cost= 0.200997487
Epoch: 0002 cost= 0.164967597
Epoch: 0003 cost= 0.145169109
Epoch: 0004 cost= 0.134419322
Epoch: 0005 cost= 0.121866673
Epoch: 0006 cost= 0.117110975
Epoch: 0007 cost= 0.115522988
Epoch: 0008 cost= 0.105438620
Epoch: 0009 cost= 0.103159308
Epoch: 0010 cost= 0.104230121
Epoch: 0011 cost= 0.100963980
Epoch: 0012 cost= 0.097668067
Epoch: 0013 cost= 0.094286390
Epoch: 0014 cost= 0.093711466
Epoch: 0015 cost= 0.092580892
Epoch: 0016 cost= 0.091860346
Epoch: 0017 cost= 0.088927798
Epoch: 0018 cost= 0.088454440
Epoch: 0019 cost= 0.086392052
Epoch: 0020 cost= 0.086117014
Optimization Finished!


'''
