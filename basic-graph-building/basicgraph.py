import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.
# the op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# create another constant that produces a 1x4 matrix
matrix3 = tf.constant([[3., 3., 3., 3.]])

# create another constant that produces a 4x1 matrix
matrix4 = tf.constant([[4.],[4.],[4.],[4.]])



# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# the returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

product2 = tf.matmul(matrix3, matrix4)

# launch the default graph. gets opened and closed using 'with' block below
# sess = tf.Session()

# to run the matmul op we call session 'run()' method, passing 'product'
# which represents the output of the matmul op. This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session. They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# the output of the op is returned in 'result' as a numpy 'ndarray' object.
# result = sess.run(product)
# print result

# close the session when we're done
# sess.close()

# closing a session with a 'with' block:
# note that the cores are specified for each computation
# in a corresponding 'with' block
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    result = sess.run([product])
    print result
  with tf.device("/cpu:1"):
    result = sess.run([product2])
    print result