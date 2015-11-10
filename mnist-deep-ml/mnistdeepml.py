import tensorflow as tf
import input_data

# load mnist data with the input_data script
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# open a session to connect to tensorflow backend
sess = tf.InteractiveSession()

# build a softmax regression model with a single linear layer

# placeholders:
# x is input images, a 2d tensor of floating point numbers
# y_ is target output classes, a 2d tensor where each row is a one-hot 10-dimensional vector
#    indicating which digit class the corresponding MNIST image belongs to.
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# weights and biases - initialized to tensors full of zeros
# W is a 784x10 matrix (because there are 784 input features and 10 outputs)
# b is a 10-dimensional vector (because we have 10 classes)
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))

# initialize all variables for use in the session
#sess.run(tf.initialize_all_variables())

# multiply vectorized input images x by the weight matrix W, add the bias b, and compute
# the softmax probabilities that are assigned to each class.
#y = tf.nn.softmax(tf.matmul(x,W) + b)

# compute cross entropy for all the images, using tf.reduce_sum  to sum across all images in the minibatch,
# as well as all classes.
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# train the model using the steepest gradient descent to descend the cross entropy
# here tensorflow adds new ops to the computation graph; e.g. computing gradients, computing parameter
# update steps, applying update steps to the parameters.
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# this op applies the gradient descent updates to the parameters. so repeatedly running
# train_step works to train the model.
#for i in range(1000):
#  batch = mnist.train.next_batch(50)
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluate the model

# tf.argmax gives you the index of the highest entry in a tensor along some axis.
# using tf.equal we check the modeled values against the real values and get a list of
# booleans, then we cast to floating point numbers and take the mean. e.g. [True, False, True, True]
# would become [1,0,1,1] which would become 0.75.
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# finally, evaluate the accuracy on the test data.
# print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# build a multilayer convolutional network

# since 0.9092 is a bad accuracy rating, we will now use a small convolutional neural network.

# weight initialization

# there need to be a lot of weights and biases, and they should be initialized with a
# small amount of noise for symmetry breaking and to prevent 0 gradients.

# we're using ReLU neurons, so it is good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons". Instead of doing this repeatedly while we build the model,
# we will make the two functions below to do it.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling

# tensorflow gives lots of flexibility in convolution and pooling operations.
# in this example, things will be kept simple. our convolutions use a stride of one and zero, padded
# so that the output is the same size as the input. our pooling is max pooling over 2x2 blocks.
# function abstraction below is done to keep code clean.

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')


# first convolutional layer

# the first layer consists of convolution followed by max pooling. the convolution will compute
# 32 features for each 5x5 patch. its weight tensor will have a shape of [5. 5. 1. 32]. The first two
# dimensions are the patch size, the next is the number of input channels, and the last is the
# number of output channels. We will also have a bias vector with a component for each output channel.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# to apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image
# width and height, and the final dimension corresponding to the number of color channels.

x_image = tf.reshape(x, [-1, 28, 28, 1])

# then we convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second convolutional layer

# in order to build a deep network, we stack several layers of this type. the second layer will have
# 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# densely connected layer

# now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neuros to allow
# processing on the entire image. we reshape the tensor from the pooling layer into a batch of vectors, multiply by
# a weight matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: to reduce overfitting, apply dropout before the readout layer. create a placeholder for the probability
# that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off
# during testing. tensorflow's tf.nn.dropout op automatically handles scaling neuron outputs in a ddition to masking
# them, so dropout just works without any additional scaling.

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# readout layer

# finally, we add a softmax layer, just like for the one layer softmax regression above.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# train and evaluate the model

# to train and evaluate the model, we can write code nearly identical to that for the simple one-layer
# SoftMax network above. The differences are: 1) we will replace the steepest gradient descent optimizer with
# more sophisticated ADAM optimizer; 2) we will include the additional parameter keep_prob in feed_dict to control
# the dropout rate; 3) and we will add logging to every 100th iteration in the training process.

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})