import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("fMNIST_data/", one_hot=True)


# softmax regression implementation

# tensorflow (specifically placeholder) lets us describe a graph of interacting
# operations that run entirely outside Python.
# 'None' means that a dimension can be of any length.
x = tf.placeholder("float", [None, 784])

# weight and bias, implemented as variables, which makes them persistent in the graph
# through all ops performed. generally, the model parameters are variables for this
# type of machine learning.
# W has a shape of [784,10] because we want to multiply the 784-dimensional image vectors by it
# to produce 10-dimensional vectors of evidence for the difference classes. b has a shape of [10]
# so we can add it to the output.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# model implementation: multiply x by W as a trick to deal with x being a 2D tensor with multiple inputs.
y = tf.nn.softmax(tf.matmul(x,W) + b)

# training of the model
# goal: to define what it means for a model to be bad, i.e. the cost or loss, and minimize that quantity.

# cross-entropy gives a good cost function.

# add a placeholder to input correct answers:
y_ = tf.placeholder("float", [None, 10])

# then implement cross entropy:
# first, 'tf.log' computes the logarithm of each element of 'y'.
# next, we multiply each element of 'y_' with the corresponding element of 'tf.log(y_)'.
# finally, 'tf.reduce_sum' adds all the elements of the tensor.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# because tensorflow knows the entire graph of computations, it can automatically use the backpropagation algorithm
# to efficiently determine how your variables affect the cost you ask it to minimize.

# here minimize cross_entropy using Gradient Descent algorithm with a learning rate of 0.01.
# in gradient descent, tensorflow shifts each variable a little bit in the direction that reduces the cost.
# tensorflow adds new operations to your graph which implement backpropagation and gradient descent. then it gives
# you back a single operation which, when run, will do a step of gradient descent training, slightly tweaking your
# variables to reduce the cost.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# now initialize the created variables
init = tf.initialize_all_variables()



