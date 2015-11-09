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

