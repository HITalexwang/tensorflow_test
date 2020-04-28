import tensorflow as tf
import functools

b = 2
l = 3
h = 4

# batch_size * seq_len * hid_size
input_tensor = tf.constant([[[3,1,2,4],
											 [7,3,5,2],
											 [1,2,6,3]],
											[[2,4,2,2],
											 [7,5,5,5],
											 [4,3,8,1]]], dtype=tf.float32)

ns = (1, l, h)
out1 = tf.nn.dropout(input_tensor, noise_shape=ns, keep_prob=0.5)
ns = (b, 1, h)
out2 = tf.nn.dropout(input_tensor, noise_shape=ns, keep_prob=0.5)
ns = (b, l, 1)
out3 = tf.nn.dropout(input_tensor, noise_shape=ns, keep_prob=0.5)

with tf.Session() as sess:
	print (sess.run(out1))
	print (sess.run(out2))
	print (sess.run(out3))