import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Create a constant op
hello = tf.constant("Hello, TensorFlow!")

# Start a TF session
sess = tf.Session()

# Run the op and get result
print(sess.run(hello))
