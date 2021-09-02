import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis Wx + b
hypothesis = x_train * W + b


"""
    t = [1., 2., 3., 4.]
    sess.run(tf.reduce_mean(t)) ==> 2.5
    tf.reduce_mean returns average value of list param
"""
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# Fit the line
for step in range(20001):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
