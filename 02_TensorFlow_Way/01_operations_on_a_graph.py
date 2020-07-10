# Operations on a Computational Graph
import numpy as np
import tensorflow as tf
import os
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

LOG_DIR = '/tmp/ML/02_TensorFlow_Way/tensorboard_logs/'

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)

# Multiplication
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs

# Add summaries to tensorboard
merged = tf.summary.merge_all()
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
# Initialize graph writer:
my_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

#writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

