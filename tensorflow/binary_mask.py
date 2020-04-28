import tensorflow as tf
import functools

#===============================================================
def unscaled_dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)*keep_prob

#===============================================================
def binary_mask(shape, keep_prob):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  ones = tf.ones(shape)
  return unscaled_dropout(ones, keep_prob)

def get(a):
	with tf.Session() as sess:
		print (sess.run(a))

# batch_size * seq_len * seq_len
layer = tf.constant([[[0,1,0],
											 [1,0,0],
											 [0,0,1]],
											[[0,0,1],
											 [0,0,1],
											 [0,1,0]]], dtype=tf.float32)

# Get the dimensions
batch_size, bucket_size, input_size = layer.get_shape().as_list()

embed_keep_prob = 0.5
# Set up the mask 
mask_shape = tf.stack([batch_size, bucket_size, 1])
mask = binary_mask(mask_shape, embed_keep_prob)

# Get the unk vector
#unk = tf.get_variable('Unk', input_size)
pred = tf.constant([[[1,0,0],
											 [0,0,1],
											 [1,0,0]],
											[[0,1,0],
											 [1,0,0],
											 [1,0,0]]], dtype=tf.float32)


#get(mask*layer)

#get((1-mask) * pred)
heads = tf.argmax(layer, -1)

s_mask = tf.squeeze(mask, -1)


layer_ = mask * layer + (1-mask) * pred

#get(layer_)

get([mask, mask*layer, (1-mask) * pred, layer_])

