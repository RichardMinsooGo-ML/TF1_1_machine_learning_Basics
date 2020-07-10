# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
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
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# Step 4. Set algorithm parameters (hyperparameters):

# Hidden_Layer_1 = 20
N_Input = 2
N_Output = 3
N_training = 1000
N_Display = 50

# Step 5. Initialize variables and placeholders:

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다.
W = tf.Variable(tf.random_uniform([N_Input, N_Output], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
# 편향은 아웃풋의 갯수, 즉 최종 결과값의 분류 갯수인 3으로 설정합니다.
b = tf.Variable(tf.zeros([N_Output]))

# Step 6. Define the model structure:

# 신경망에 가중치 W과 편향 b을 적용합니다
L = tf.add(tf.matmul(X, W), b)
# 가중치와 편향을 이용해 계산한 결과 값에
# 텐서플로우에서 기본적으로 제공하는 활성화 함수인 ReLU 함수를 적용합니다.
L = tf.nn.relu(L)

# 마지막으로 softmax 함수를 이용하여 출력값을 사용하기 쉽게 만듭니다
# softmax 함수는 다음처럼 결과값을 전체합이 1인 확률로 만들어주는 함수입니다.
# 예) [8.04, 2.76, -6.52] -> [0.53 0.24 0.23]
model = tf.nn.softmax(L)

# Step 7. Declare the loss functions and Optimizer:

# 신경망을 최적화하기 위한 비용 함수를 작성합니다.
# 각 개별 결과에 대한 합을 구한 뒤 평균을 내는 방식을 사용합니다.
# 전체 합이 아닌, 개별 결과를 구한 뒤 평균을 내는 방식을 사용하기 위해 axis 옵션을 사용합니다.
# axis 옵션이 없으면 -1.09 처럼 총합인 스칼라값으로 출력됩니다.
#        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
# 예) [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
#     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]
# 즉, 이것은 예측값과 실제값 사이의 확률 분포의 차이를 비용으로 계산한 것이며,
# 이것을 Cross-Entropy 라고 합니다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# Step 8. Initialize and train the model:
# 그래프를 실행할 세션을 구성합니다.
# sess = tf.Session()

###################
# 신경망 모델 학습
###################

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#for step in range(100):
for step in range(N_training):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % N_Display == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
# Step 9. Evaluate the model:

#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('Prediction Value:', sess.run(prediction, feed_dict={X: x_data}))
print('Actual Value:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

# 세션을 닫습니다.
# sess.close()
sess.close()

# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:


