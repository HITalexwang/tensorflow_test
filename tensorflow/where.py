import tensorflow as tf
import functools

allowed_head_cnt = tf.constant([0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=tf.int32)
null_index_tensor = tf.constant([8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=tf.int32)
selected_heads_ = tf.constant([0, 2, 3, 7, 6, 6, 3, 0, 0], dtype=tf.int32)

# must use tf.equal, not == !
selected_gold_heads = tf.where(tf.equal(allowed_head_cnt,0), null_index_tensor, selected_heads_)

with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  #print (sess.run(output))
  print (sess.run(selected_gold_heads))
