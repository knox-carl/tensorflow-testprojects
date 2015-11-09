import tensorflow as tf

# the feed mechanism patches a tensor directly into an operation in the graph
# a feed temporarily replaces the output of an operation with a tensor value. you
# supply feed data as an argument to a run() call. the feed is only used for the run call to which it is
# passed. the most common use case involves designating specific operations to be "feed" operations by using
# tf.placeholder() to create them, as shown below.

input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

# a placeholder() operation generates an error if you do not supply a feed for it.

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7,], input2:[2.]})


# a much larger scale example of feeds is hosted here:
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py