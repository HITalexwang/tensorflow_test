import tensorflow as tf

max_k = 4
batch_size = 3

# [batch_size]
a = tf.random_uniform([batch_size], minval=0, maxval=max_k, dtype=tf.int32)

a_ = tf.tile(tf.expand_dims(a, -1), [1, max_k])

r = tf.expand_dims(tf.range(max_k), 0)

# [batch_size, max_k]
r_ = tf.tile(r, [batch_size, 1])

one = tf.ones_like(r_)
zero = tf.zeros_like(r_)

values = tf.where(r_ < a_, one, zero)

with tf.Session() as sess:
  print (sess.run([a_, r_, values]))
