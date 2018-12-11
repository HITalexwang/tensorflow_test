# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
#import modeling
#import optimization
#import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the outputs will be written.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_export", False, "Whether to export the model.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("num_epochs_per_eval", 1.0,
                   "Number of training epochs to run between evaluations.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


import numpy as np

def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0

def get_y(x):
  return 10 + x*x


def sample_data(n=10000, scale=100):
  data = []

  x = scale*(np.random.random_sample((n,))-0.5)

  for i in range(n):
    yi = get_y(x[i])
    data.append([x[i], yi])

  return np.array(data)

def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])

def generator(Z,hsize=[16, 16],reuse=False):
  with tf.variable_scope("GAN/Generator",reuse=reuse):
    h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
    h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
    out = tf.layers.dense(h2,2)

  return out

def discriminator(X,hsize=[16, 16],reuse=False):
  with tf.variable_scope("GAN/Discriminator",reuse=reuse):
    h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
    h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
    h3 = tf.layers.dense(h2,2)
    out = tf.layers.dense(h3,1)

  return out, h3

def model_fn_builder(init_checkpoint, learning_rate, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    #tf.logging.info("*** Features ***")
    #for name in sorted(features.keys()):
    #  tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    z = features["z"]
    G_sample = generator(z)
    f_logits, g_rep = discriminator(G_sample)
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
      x = features["x"]
      r_logits, r_rep = discriminator(x,reuse=True)
      disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    scaffold_fn = None
    """
    tvars = tf.trainable_variables()
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
      disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

      global_step = tf.train.get_or_create_global_step()
      new_global_step = global_step + 1

      gen_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(gen_loss,var_list = gen_vars) # G Train step
      disc_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(disc_loss,var_list = disc_vars) # D Train step

      train_op = tf.group(gen_step, disc_step, [global_step.assign(new_global_step)])
      #train_op = optimization.create_optimizer(
          #total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=gen_loss+disc_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(G_sample):
        false_pred = tf.where(f_logits>0, tf.ones_like(f_logits), tf.zeros_like(f_logits))
        true_pred = tf.where(r_logits>0, tf.ones_like(r_logits), tf.zeros_like(r_logits))
        fp = tf.metrics.accuracy(tf.ones_like(f_logits), false_pred)
        tp = tf.metrics.accuracy(tf.ones_like(r_logits), true_pred)
        y = G_sample[:, 0] * G_sample[:, 0] + 10
        dist = tf.losses.mean_squared_error(y, G_sample[:,1])
        return {
            "gen_loss": tf.metrics.mean(gen_loss),
            "disc_loss": tf.metrics.mean(disc_loss),
            "false_positive": fp,
            "true_positive": tp,
            "dist": tf.metrics.mean(dist),
        }

      eval_metrics = (metric_fn, [G_sample])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=gen_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=f_logits, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(x, z, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  assert len(x) == len(z)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(x)
    seq_length = 2

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "x":
            tf.constant(
                x, shape=[num_examples, seq_length],
                dtype=tf.float32),
        "z":
            tf.constant(
                z,
                shape=[num_examples, seq_length],
                dtype=tf.float32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def pred_input_fn_builder(z, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(z)
    seq_length = 2

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "z":
            tf.constant(
                z,
                shape=[num_examples, seq_length],
                dtype=tf.float32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def build_tensor_serving_input_receiver_fn(shape, batch_size=1, dtype=tf.float32):


  def serving_input_receiver_fn():
    # Prep a placeholder where the input example will be fed in
    features = {
    'z':tf.placeholder(dtype=dtype, shape=[batch_size] + shape, name='z'),
    'x':tf.placeholder(dtype=dtype, shape=[batch_size] + shape, name='x')}

    return tf.estimator.export.ServingInputReceiver(
      features=features, receiver_tensors=features)

  return serving_input_receiver_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  #bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_xs = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    n_train = 256
    train_xs = sample_data(n=n_train)
    train_zs = sample_Z(m=n_train, n=2)
    train_steps_per_epoch = int(len(train_xs) / FLAGS.train_batch_size)
    train_steps_per_eval = int(
        FLAGS.num_epochs_per_eval * train_steps_per_epoch)
    #num_train_steps = int(len(train_xs) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    #num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  if FLAGS.do_eval:
    n_eval = 64
    eval_xs = sample_data(n=n_eval)
    eval_zs = sample_Z(m=n_eval, n=2)

    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_xs) / FLAGS.eval_batch_size)


  model_fn = model_fn_builder(
      #bert_config=bert_config,
      #num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      #num_train_steps=num_train_steps,
      #num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train and FLAGS.do_eval:
    tf.logging.info("***** Training data *****")
    tf.logging.info("  Num examples = %d", len(train_xs))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num epochs = %d", FLAGS.num_train_epochs)
    tf.logging.info("  Steps per eval = %d", train_steps_per_eval)
    tf.logging.info("  Steps per epoch = %d", train_steps_per_epoch)
    tf.logging.info("***** Evaluation data *****")
    tf.logging.info("  Num examples = %d", len(eval_xs))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    train_input_fn = input_fn_builder(
        x=train_xs, z=train_zs,
        is_training=True,
        drop_remainder=True)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = input_fn_builder(
        x=eval_xs, z=eval_zs,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    best_model_path = os.path.join(FLAGS.model_dir, 'best_model')
    if not os.path.exists(best_model_path):
      os.makedirs(best_model_path)
    current_step = _load_global_step_from_checkpoint_dir(FLAGS.model_dir)
    #print (current_step)
    best_dist = 1e9
    while current_step < FLAGS.num_train_epochs * train_steps_per_epoch:
      estimator.train(input_fn=train_input_fn, steps=train_steps_per_eval)
      current_step += train_steps_per_eval
      tf.logging.info('Starting evaluation at step=%d.' % current_step)
      result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      output_eval_file = os.path.join(FLAGS.model_dir, "eval_results.txt")
      with tf.gfile.GFile(output_eval_file, "a") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))
        if result["dist"] < best_dist:
          best_dist = result["dist"]
          tf.logging.info("  New record: {:.4f}, saving model.".format(result["dist"]))
          writer.write("New record: {:.4f}, saving model.\n".format(result["dist"]))
          cmd = "rm " + best_model_path + "/*"
          cmd += ";cp " + os.path.join(FLAGS.model_dir, "model.ckpt-"+str(current_step)) + "* " + best_model_path
          print (cmd)
          os.system(cmd)
        writer.write("\n")

    if FLAGS.do_export:
      tf.logging.info('Starting exporting saved model ...')
      estimator.export_saved_model(
        export_dir_base=FLAGS.model_dir + '/export_savedmodel/',
        serving_input_receiver_fn=build_tensor_serving_input_receiver_fn(
            [2], batch_size=FLAGS.predict_batch_size), as_text=True)

  """
  if FLAGS.do_train:
    
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_xs))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = input_fn_builder(
        x=train_xs, z=train_zs,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  """

  """
  if FLAGS.do_eval:
    n_eval = 24
    eval_xs = sample_data(n=n_eval)
    eval_zs = sample_Z(m=n_eval, n=2)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_xs))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_xs) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = input_fn_builder(
        x=eval_xs, z=eval_zs,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
  """

  if FLAGS.do_predict:
    n_pred = 32
    predict_zs = sample_Z(m=n_pred, n=2)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_zs))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = pred_input_fn_builder(
        z=predict_zs,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(
            str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)


if __name__ == "__main__":
  #flags.mark_flag_as_required("data_dir")
  #flags.mark_flag_as_required("vocab_file")
  #flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("model_dir")
  tf.app.run()
