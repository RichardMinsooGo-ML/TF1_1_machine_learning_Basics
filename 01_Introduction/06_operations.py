# Operations
#----------------------------------
#
# This function introduces various operations
# in TensorFlow

# Declaring Operations
import tensorflow as tf
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# div() vs truediv() vs floordiv()
print("-------------------------------------------------------------------------------------")
print("div(3,4) div() returns the same type as the inputs")
print(sess.run(tf.div(3, 4)))
print("-------------------------------------------------------------------------------------")
print("truediv(3,4) which casts integers into floats before dividing and always returning a float")
print(sess.run(tf.truediv(3, 4)))
print("-------------------------------------------------------------------------------------")
print("floordiv(3,4) this will still return a float, but rounded down to the nearest integer")
print(sess.run(tf.floordiv(3.0, 4.0)))

# Mod function
print("-------------------------------------------------------------------------------------")
print("Mod function(22.0, 5.0) returns the remainder after the division.")
print(sess.run(tf.mod(22.0, 5.0)))

# Cross Product
print("-------------------------------------------------------------------------------------")
print("The cross-product is only defined for two three-dimensional vectors")
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))

# Trig functions
print("-------------------------------------------------------------------------------------")
print("sin(Pi)")
print(sess.run(tf.sin(3.1416)))
print("-------------------------------------------------------------------------------------")
print("cos(Pi)")
print(sess.run(tf.cos(3.1416)))
print("-------------------------------------------------------------------------------------")
print("tangent(Pi/4)")      
print(sess.run(tf.tan(3.1416/4.)))

# Custom operation
test_nums = range(15)


def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return tf.subtract(3 * tf.square(x_val), x_val) + 10
print("-------------------------------------------------------------------------------------")
print("Return 3x^2 - x + 10, x=11")    
print(sess.run(custom_polynomial(11)))

# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print("-------------------------------------------------------------------------------------")
print("Expected output Return 3x^2 - x + 10") 
print(expected_output)

print("-------------------------------------------------------------------------------------")
print("TensorFlow custom function output 3x^2 - x + 10") 

# TensorFlow custom function output
for num in test_nums:
    print(sess.run(custom_polynomial(num)))

    
    