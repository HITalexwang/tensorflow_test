import tensorflow as tf
import functools

def get(a):
	with tf.Session() as sess:
		print (sess.run(a))


layer_embedding = tf.constant([10,10,10,10], dtype=tf.int32)

# batch_size * seq_len * hid_size
input_tensor = tf.constant([[[0,1,0,4],
											 [7,3,5,2],
											 [1,2,6,3]],
											[[2,4,2,2],
											 [7,5,5,5],
											 [0,3,8,1]]], dtype=tf.int32)

hid_size = 4
layer_embedding = tf.reshape(layer_embedding, [1,1,hid_size])

get(input_tensor+layer_embedding)

exit()

# batch_size * seq_len * seq_len
mask = tf.constant([[[0,1,0],
											 [0,1,0],
											 [0,1,0]],
											[[0,1,1],
											 [0,1,1],
											 [0,1,1]]], dtype=tf.int32)
# batch_size * seq_len
null_mask = tf.constant([[1,0,0],[1,0,0]], dtype=tf.int32)

# batch_size * 1 * seq_len
null_mask_ = tf.expand_dims(null_mask, axis=1)

add = mask + null_mask_


#get(add)

heads = tf.constant([[[0,1,0],
											 [0,0,1],
											 [0,1,1]],
											[[0,0,0],
											 [0,1,0],
											 [0,0,0]]], dtype=tf.int32)

zeros = tf.zeros_like(heads)
# batch * seq_len
cnt = tf.reduce_sum(heads, axis=-1, keep_dims=True)

cnt_ = cnt + zeros

null_mask = tf.constant([[1,0,0],[1,0,0]], dtype=tf.int32)


null_mask_ = tf.expand_dims(null_mask, axis=1)

expanded_null_mask = null_mask_ + zeros

a = tf.where(tf.equal(cnt_, 0), expanded_null_mask, heads)


get(cnt_)

get(a)