import tensorflow as tf
import functools

# batch_size * seq_len
indices = tf.constant([[1,0,2],[0,2,1]], dtype=tf.int32)

depth = 3
# batch_size * seq_len * seq_len
matrix = tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)

with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  #print (sess.run(output))
  print (sess.run(matrix))

"""
indices = tf.constant([[1,0], [1,1], [0,1]])
updates = tf.constant([2, 3, 0])
shape = tf.constant([2,2])
# batch_size * seq_len * seq_len
matrix = tf.scatter_nd(indices, updates, shape)

with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  #print (sess.run(output))
  print (sess.run(matrix))
"""