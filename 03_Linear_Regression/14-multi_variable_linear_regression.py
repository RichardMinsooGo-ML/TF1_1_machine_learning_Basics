# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
print(hypothesis)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

'''
0 Cost:  19614.8
Prediction:
 [ 21.69748688  39.10213089  31.82624626  35.14236832  32.55316544]
10 Cost:  14.0682
Prediction:
 [ 145.56100464  187.94958496  178.50236511  194.86721802  146.08096313]

 ...

1990 Cost:  4.9197
Prediction:
 [ 148.15084839  186.88632202  179.6293335   195.81796265  144.46044922]
2000 Cost:  4.89449
Prediction:
 [ 148.15931702  186.8805542   179.63194275  195.81971741  144.45298767]
'''
