# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

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

# Declaring matrices
sess = tf.Session()

# Declaring matrices

# Identity matrix
identity_matrix = tf.diag([1.0,1.0,1.0])
print("Identity Matrix")
print(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
print("------------------------------------------------------------------------")
print("2x3 random norm matrix A")
print(sess.run(A))

# 2x3 constant matrix
B = tf.fill([2,3], 5.0)
print("------------------------------------------------------------------------")
print("2x3 constant matrix B")
print(sess.run(B))

# 3x2 random uniform matrix
C = tf.random_uniform([3,2])
print("------------------------------------------------------------------------")
print("3x2 random uniform matrix C")
print(sess.run(C))  # Note that we are reinitializing, hence the new random variables

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print("------------------------------------------------------------------------")
print("Create matrix from np array D")
print(sess.run(D))

# Matrix addition/subtraction
print("------------------------------------------------------------------------")
print("Matrix addition A+B")
print(sess.run(A+B))
print("------------------------------------------------------------------------")
print("Matrix subtraction B-B")
print(sess.run(B-B))


# Matrix Multiplication
print("------------------------------------------------------------------------")
print("Matrix Multiplication with B and Identity Matrix")
print(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose
print("------------------------------------------------------------------------")
print("Matrix Transpose from C")
print(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant
print("------------------------------------------------------------------------")
print("Matrix Determinant from D")
print(sess.run(tf.matrix_determinant(D)))

# Matrix Inverse
print("------------------------------------------------------------------------")
print("Matrix Inverse from D")
print(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
print("------------------------------------------------------------------------")
print("Cholesky Decomposition from Identity Matrix")
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors
print("------------------------------------------------------------------------")
print("Eigenvalues and Eigenvectors from D")
print(sess.run(tf.self_adjoint_eig(D)))


