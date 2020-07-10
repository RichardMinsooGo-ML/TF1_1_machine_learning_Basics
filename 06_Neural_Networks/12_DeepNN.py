# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
# 신경망의 레이어를 여러개로 구성하여 말로만 듣던 딥러닝을 구성해 봅시다!
import tensorflow as tf
import numpy as np

# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Step 1. Import or generate datasets:
# Step 2. Transform and normalize data:
# Step 3. Partition datasets into train, test, and validation sets:

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# Step 4. Set algorithm parameters (hyperparameters):

Hidden_Layer_1 = 20
N_input = 2
N_Output = 3
N_training = 10001
N_Display = 50

# Step 5. Initialize variables and placeholders:

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수] -> [N_Input, Hidden_Layer_1] 으로 정합니다.
W1 = tf.Variable(tf.random_uniform([N_input, Hidden_Layer_1], -1., 1.))

# 두번째 가중치의 차원을 [첫번째 히든 레이어의 뉴런 갯수, 분류 갯수] -> [Hidden_Layer_1, 3] 으로 정합니다.
W2 = tf.Variable(tf.random_uniform([Hidden_Layer_1, N_Output], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
# b1 은 히든 레이어의 뉴런 갯수로, b2 는 최종 결과값 즉, 분류 갯수인 3으로 설정합니다.

b1 = tf.Variable(tf.zeros([Hidden_Layer_1]))
b2 = tf.Variable(tf.zeros([N_Output]))

# Step 6. Define the model structure:

# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 최종적인 아웃풋을 계산합니다.
# 히든레이어에 두번째 가중치 W2와 편향 b2를 적용하여 3개의 출력값을 만들어냅니다.
model = tf.add(tf.matmul(L1, W2), b2)

# Step 7. Declare the loss functions and Optimizer:

# 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
# 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# Step 8. Initialize and train the model:
# 그래프를 실행할 세션을 구성합니다.
# sess = tf.Session()

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(N_training):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % N_Display == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# Step 9. Evaluate the model:
        
#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print('Predicted Value:', sess.run(prediction, feed_dict={X: x_data}))
print('Actual Value:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

# 세션을 닫습니다.
# sess.close()
sess.close()

# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:



