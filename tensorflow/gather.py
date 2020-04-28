import tensorflow as tf

def get(a):
	with tf.Session() as sess:
		print (sess.run(a))

batch_size = 2
rel_size = 3
seq_len = 2
hid_size = 3
# batch_size (2) * seq_len (2) * rel_size (3) * hid_size (3)
layer_rel = tf.constant([[[[3,2,1],
											 [4,5,6],
											 [7,8,9]],
											[[1,4,2],
											 [2,3,6],
											 [3,8,6]]],

										 [[[2,8,2],
											 [4,9,6],
											 [1,2,9]],
											[[3,4,2],
											 [2,9,6],
											 [3,2,5]]]], dtype=tf.float32)

# batch_size (2) * seq_len (2) * seq_len (2)

rel_pred = tf.constant([[[0,2],
											 [1,0]],
											[[0,0],
											 [2,0]]], dtype=tf.int32)

arc_pred = tf.constant([[[0,1],
											 [1,0]],
											[[0,0],
											 [1,0]]], dtype=tf.int32)

rel_onehot = tf.one_hot(rel_pred, 3, on_value=1.0, off_value=0.0, axis=-1)

#get(rel_onehot)

# batch_size (2) * seq_len (2) * seq_len (2) * rel_size (3)
masked_rel_onehot = rel_onehot * tf.to_float(tf.expand_dims(arc_pred,-1))

#get(masked_rel_onehot)

layer_rel = tf.reshape(layer_rel, tf.stack([-1,rel_size,hid_size]))

masked_rel_onehot = tf.reshape(masked_rel_onehot, tf.stack([-1,seq_len,rel_size]))

# (batch_size * seq_len) x seq_len x hid_size
multi = tf.matmul(masked_rel_onehot, layer_rel)

get(multi)

exp = tf.reshape(multi, tf.stack([batch_size, seq_len, seq_len, hid_size]))

get(exp)

get(tf.reduce_sum(exp, axis=-2))

# equals to the previous way
get(tf.reshape(tf.reduce_sum(multi, axis=-2), tf.stack([batch_size, seq_len, hid_size])))

exit()



# batch_size * seq_len
rel_ = tf.reduce_sum(rel_pred, axis=-1)

# batch_size * seq_len * rel_size
rel_onehot = tf.one_hot(rel_, 3, on_value=1, off_value=0, axis=-1)

rel_onehot_ = tf.to_float(tf.expand_dims(rel_onehot, axis=-1))

masked_rel_layer = rel_onehot_ * layer_rel

rel_tensor = tf.reduce_sum(masked_rel_layer, axis=-2)

get(rel_onehot)

get(masked_rel_layer)

get(rel_tensor)

