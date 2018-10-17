import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses.losses_impl import Reduction, compute_weighted_loss

def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None, fp_rate=1.0, fn_rate=1.0,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.
  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.
  If `label_smoothing` is nonzero, smooth the labels towards 1/2:
      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing
  Args:
    multi_class_labels: `[batch_size, num_classes]` target integer labels in
      `{0, 1}`.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    label_smoothing: If greater than `0` then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `logits`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weights` is invalid, or if
      `weights` is None.  Also if `multi_class_labels` or `logits` is None.
  @compatbility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if multi_class_labels is None:
    raise ValueError("multi_class_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      (logits, multi_class_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    if label_smoothing > 0:
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses, fn_cost, fp_cost = sigmoid_cross_entropy_with_logits(labels=multi_class_labels,
                                                  logits=logits,
                                                  name="xentropy",
                                                  fp_rate=fp_rate,
                                                  fn_rate=fn_rate)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction), fn_cost, fp_cost


def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None,
    fp_rate=None,
    fn_rate=None):
  """Computes sigmoid cross entropy given `logits`.
  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.
  For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))
  For x < 0, to avoid overflow in exp(-x), we reformulate the above
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))
  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation
      max(x, 0) - x * z + log(1 + exp(-abs(x)))
  `logits` and `labels` must have the same type and shape.
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                           labels, logits)
  # pylint: enable=protected-access

  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    ones = array_ops.ones_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    #print (cond)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    fn_cond = math_ops.logical_and(labels > zeros, logits < zeros)
    fp_cond = math_ops.logical_and(labels <= zeros, logits >= zeros)
    fn_cost = fn_rate * math_ops.cast(fn_cond, dtypes.float32)
    fp_cost = fp_rate * math_ops.cast(fp_cond, dtypes.float32)

    pos_loss = logits - logits * labels + labels * math_ops.log1p(math_ops.exp(-logits + fn_cost)) + (
    	(ones - labels) * math_ops.log1p(math_ops.exp(-logits + fp_cost)))
    neg_loss = - logits * labels + labels * math_ops.log(math_ops.exp(logits) + math_ops.exp(fn_cost)) + (
    	(ones - labels) * math_ops.log(math_ops.exp(logits) + math_ops.exp(fp_cost)))

    return array_ops.where(cond, pos_loss, neg_loss, name=name), fn_cost, fp_cost


if __name__ == '__main__':
	labels = tf.constant([0,1,1], dtype=tf.float32)
	logits = tf.constant([10,-3,5], dtype=tf.float32) # [1,0,1]
	loss, fn_cost, fp_cost = sigmoid_cross_entropy(labels, logits, fp_rate=.5, fn_rate=.4)
	with tf.Session() as sess:
		print (sess.run([loss, fn_cost, fp_cost]))