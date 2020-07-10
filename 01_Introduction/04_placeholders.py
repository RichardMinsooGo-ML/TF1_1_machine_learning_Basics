# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow

import numpy as np
import tensorflow as tf

# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Introduce tensors in tf


# Using Placeholders
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
# Input data to placeholder, note that 'rand_array' and 'x' are the same shape.
rand_array = np.random.rand(4, 4)

# Create a Tensor to perform an operation (here, y will be equal to x, a 4x4 matrix)
y = tf.identity(x)

# Print the output, feeding the value of x into the computational graph

print(sess.run(y, feed_dict={x: rand_array}))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph)

