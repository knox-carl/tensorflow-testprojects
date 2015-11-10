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
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initialize all variables for use in the session
sess.run(tf.initialize_all_variables())

# multiply vectorized input images x by the weight matrix W, add the bias b, and compute
# the softmax probabilities that are assigned to each class.
y = tf.nn.softmax(tf.matmul(x,W) + b)

# compute cross entropy for all the images, using tf.reduce_sum  to sum across all images in the minibatch,
# as well as all classes.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# train the model using the steepest gradient descent to descend the cross entropy
# here tensorflow adds new ops to the computation graph; e.g. computing gradients, computing parameter
# update steps, applying update steps to the parameters.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# this op applies the gradient descent updates to the parameters. so repeatedly running
# train_step works to train the model.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluate the model

# tf.argmax gives you the index of the highest entry in a tensor along some axis.
# using tf.equal we check the modeled values against the real values and get a list of
# booleans, then we cast to floating point numbers and take the mean. e.g. [True, False, True, True]
# would become [1,0,1,1] which would become 0.75.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# finally, evaluate the accuracy on the test data.
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# build a multilayer convolutional network

# since 0.9092 is a bad accuracy rating, we will now use a small convolutional neural network.

