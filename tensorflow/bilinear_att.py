import tensorflow as tf
import six

#***************************************************************
def get_sizes(t):
  """"""
  
  shape = []
  for i in six.moves.range(len(t.get_shape().as_list()[:-1])):
    shape.append(tf.shape(t)[i])
  shape.append(t.get_shape().as_list()[-1])
  return shape

#===============================================================
def reshape(t, shape):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.reshape(t, shape)

#===============================================================
def dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)

#===============================================================
def bilinear_attention(layer1, layer2, hidden_keep_prob=.2, add_linear=True):
  """"""
  
  layer_shape = get_sizes(layer1)
  bucket_size = layer_shape[-2]
  input1_size = layer_shape.pop()+add_linear
  input2_size = layer2.get_shape().as_list()[-1]
  ones_shape = tf.stack(layer_shape + [1])
  
  weights = tf.get_variable('Weights', shape=[input1_size, input2_size], initializer=tf.zeros_initializer)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
  original_layer1 = layer1
  if hidden_keep_prob < 1.:
    noise_shape1 = tf.stack(layer_shape[:-1] + [1, input1_size-add_linear])
    noise_shape2 = tf.stack(layer_shape[:-2] + [1, input2_size])
    layer1 = dropout(layer1, hidden_keep_prob, noise_shape=noise_shape1)
    layer2 = dropout(layer2, hidden_keep_prob, noise_shape=noise_shape2)
  if add_linear:
    ones = tf.ones(ones_shape)
    layer1 = tf.concat([layer1, ones], -1)

  # (n x m x d) -> (nm x d)
  layer1 = reshape(layer1, [-1, input1_size])
  # (n x m x d) -> (n x m x d)
  layer2 = reshape(layer2, [-1, bucket_size, input2_size]) # [b*n, seq, 512]
  
  # (nm x d) * (d x d) -> (nm x d)
  attn = tf.matmul(layer1, weights)
  # (nm x d) -> (n x m x d)
  attn = reshape(attn, [-1, bucket_size, input2_size])

  #print (attn.get_shape(), layer2.get_shape())
  att_shape = tf.shape(layer2)
  # (n x m x d) * (n x m x d) -> (n x m x m)
  attn = tf.matmul(attn, layer2, transpose_b=True)
  # (n x m x m) -> (n x m x m)
  attn = reshape(attn, layer_shape + [bucket_size])
  # (n x m x m) -> (n x m x m)
  soft_attn = tf.nn.softmax(attn)
  # (n x m x m) * (n x m x d) -> (n x m x d)
  weighted_layer1 = tf.matmul(soft_attn, original_layer1)
  
  return attn, weighted_layer1, att_shape

# b=2, n=1, seq=3, h=4 
l1 = tf.constant([[[1,2,3,4],[3,4,5,6],[7,8,9,0]],[[4,3,2,1],[5,4,7,1],[8,4,6,2]]], dtype=tf.float32)

l2 = tf.constant([[[5,4,7,1],[3,4,5,6],[8,9,1,0]],[[3,4,5,6],[4,3,2,1],[5,4,7,1]]], dtype=tf.float32)

att, _, shape = bilinear_attention(l1, l2)

print (shape)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print (sess.run(shape))
