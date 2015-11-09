import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("fMNIST_data/", one_hot=True)

# tensorflow (specifically placeholder) lets us describe a graph of interacting
# operations that run entirely outside Python.
# 'None' means that a dimension can be of any length.
x = tf.placeholder("float", [None, 784])

# weight and bias, implemented as variables, which makes them persistent in the graph
# through all ops performed. generally, the model parameters are variables for this
# type of machine learning.
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
