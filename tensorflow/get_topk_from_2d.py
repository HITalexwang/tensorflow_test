import tensorflow as tf
import functools

def get(a):
	with tf.Session() as sess:
		print (sess.run(a))

# batch_size * seq_len * seq_len
scores = tf.constant([[[1,3,2],
											 [5,2,1],
											 [0,2,7]],
											[[1,0,2],
											 [0,2,9],
											 [3,7,1]]], dtype=tf.int32)

batch_size = 2
to_seq_len = 3
k = 3

_, top_indices = tf.nn.top_k(tf.reshape(scores, (batch_size,-1)), k)
# batch_size * k * 2
top_indices = tf.stack(((top_indices // to_seq_len), (top_indices % to_seq_len)), -1)

batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
batch_index = tf.tile(batch_idx, [1, k, 1])

batched_indices = tf.concat([batch_index, top_indices], axis=-1)

reshaped_indices = tf.reshape(batched_indices, [-1, 3])

# batch_size * k
updates = tf.ones(batch_size*k)
shape = tf.constant([batch_size, to_seq_len, to_seq_len])
matrix = tf.scatter_nd(reshaped_indices, updates, shape)
# batch_size * seq_len

#with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  #print (sess.run(output))
#  print (sess.run(top_indices))
#  print (sess.run(matrix))

get(top_indices)

debug = False
if debug:
	get(batch_index)
	get(batched_indices)
	get(reshaped_indices)
	get(shape)
	get(updates)

get(matrix)

