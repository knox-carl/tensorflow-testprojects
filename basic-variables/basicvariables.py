import tensorflow as tf

# Create a Variable that will be initialized to the scalar value 0
var = tf.Variable(0, name="counter")

# Create an op to add one to 'var'.

one = tf.constant(1)
new_value = tf.add(var, one)
update = tf.assign(var, new_value)

# Variables must be initialized by running an 'init' op after having
# launched the graph. we first have to add the 'init' op to the graph.
init_op = tf.initialize_all_variables()

# launch the graph and run the ops.
with tf.Session() as sess:
    # run the 'init' op
    sess.run(init_op)
    # print the initial value of 'var'
    print sess.run(var)
    # run the op that updates 'var' and prints 'var'
    for _ in range(3):
        sess.run(update)
        print sess.run(var)


