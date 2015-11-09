import tensorflow as tf

# to fetch the outputs of ops, execute the graph with a run() call on the Session object and pass in
# the tensors to retrieve.

# below it is shown how to fetch multiple tensors (e.g. mul and intermed)

# all the ops needed to produce the values of the requested tensors are run once ( not once per requested
# tensor )

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print result