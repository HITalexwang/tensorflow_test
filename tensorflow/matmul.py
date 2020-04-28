import tensorflow as tf
import functools

def get(a):
	with tf.Session() as sess:
		print (sess.run(a))

b = 2
s = 3
h = 4

# batch_size * seq_len * hid_size
input_tensor = tf.constant([[[0,1,0,4],
											 [7,3,5,2],
											 [1,2,6,3]],
											[[2,4,2,2],
											 [7,5,5,5],
											 [0,3,8,1]]], dtype=tf.int32)

# batch_size * seq_len * 1
weight = tf.constant([[[0],
											 [1],
											 [3]],
											[[1],
											 [2],
											 [4]]], dtype=tf.int32)

weight_ = tf.reshape(weight, [b, 1, s])

mul = tf.matmul(weight_, input_tensor)

ma = tf.squeeze(mul, 1)

mb = tf.reshape(mul, [b, h])

get(weight_)

get(mul)

get(ma)

get(mb)