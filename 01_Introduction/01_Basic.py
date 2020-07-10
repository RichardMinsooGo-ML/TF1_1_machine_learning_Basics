# 텐서플로우의 기본적인 구성을 익힙니다.
import tensorflow as tf

# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# tf.constant: 말 그대로 상수입니다.
hello = tf.constant('Hello, TensorFlow!')
print("hello = ",hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)  # a + b 로도 쓸 수 있음
print("a = ",a)
print("b = ",b)
print("c = ",c)


# 위에서 변수와 수식들을 정의했지만, 실행이 정의한 시점에서 실행되는 것은 아닙니다.
# 다음처럼 Session 객제와 run 메소드를 사용할 때 계산이 됩니다.
# 따라서 모델을 구성하는 것과, 실행하는 것을 분리하여 프로그램을 깔끔하게 작성할 수 있습니다.
# 그래프를 실행할 세션을 구성합니다.
sess = tf.Session()
# sess.run: 설정한 텐서 그래프(변수나 수식 등등)를 실행합니다.

hello_run = sess.run(hello)
a_run = sess.run(a)
b_run = sess.run(b)
c_run = sess.run(c)

print("Session-run hello = ",hello_run)

#print(sess.run([a, b, c]))
print("Session-run a = ",a_run)
print("Session-run b = ",b_run)
print("Session-run c(=a+b) = ",c_run)

# 세션을 닫습니다.
sess.close()
