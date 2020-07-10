import tensorflow as tf
import numpy as np

# This is used only for High level of CPU.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Step 1. Import or generate datasets:
# Step 2. Transform and normalize data:
# Step 3. Partition datasets into train, test, and validation sets:

tf.set_random_seed(777)  # for reproducibility

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

# Step 4. Set algorithm parameters (hyperparameters):
learning_rate = 0.001
N_Display = 500
N_training_Steps = 10000

# Step 5. Initialize variables and placeholders:

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Step 6. Define the model structure:

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
Pred_m = tf.sigmoid(tf.matmul(X, W) + b)

# Step 7. Declare the loss functions:

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(Pred_m) + (1 - Y) *
                       tf.log(1 - Pred_m))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if Pred_m>0.5 else False
predicted = tf.cast(Pred_m > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Step 8. Initialize and train the model:
# 그래프를 실행할 세션을 구성합니다.
# sess = tf.Session()

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(N_training_Steps):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if (step + 1) % N_Display == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run(W))

    # Accuracy report
    h, c, a = sess.run([Pred_m, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

# Step 9. Evaluate the model:
# 세션을 닫습니다.
sess.close()

# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:
