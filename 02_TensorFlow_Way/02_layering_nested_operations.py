# Layering Nested Operations

import os
import numpy as np
import tensorflow as tf
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

LOG_DIR = '/tmp/ML/02_TensorFlow_Way/tensorboard_logs/'

# Start a graph session
sess = tf.Session()

# Create the data and variables
my_array = np.array([[1., 3., 5., 7., 9.],
                   [-2., 0., 2., 4., 6.],
                   [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

# Constants for matrix multiplication:
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# Create our multiple operations
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

print("my_array = ",my_array)
print("x_vals = ",x_vals)

# Now feed data through placeholder and print results
for x_val in x_vals:
    print("------------------")
    print(sess.run(add1, feed_dict={x_data: x_val}))


# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs

# Add summaries to tensorboard
merged = tf.summary.merge_all()
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
# Initialize graph writer:
my_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

