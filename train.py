"""Training Script"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os.path
import sys

import tensorflow as tf
import numpy as np

from utils import DataLoader
from models.sketchanet import SketchANet

# Basic model parameters as external flags.
FLAGS = None

def training():
  data_loader = DataLoader(batch_size=FLAGS.batch_size)
  model = SketchANet(learning_rate=FLAGS.learning_rate)

  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, name='images_placeholder', shape=(FLAGS.batch_size, 225, 225, 1))
    labels_placeholder = tf.placeholder(tf.int32, name='labels_placeholder', shape=(FLAGS.batch_size))

    logits = model.inference(images_placeholder)
    loss = model.loss(logits, labels_placeholder)
    train_op = model.training(loss)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)

    for step in xrange(FLAGS.max_steps):
      images, labels = data_loader.next_batch()
      feed_dict = { images_placeholder: images, labels_placeholder: labels }
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      if step % 100 == 0:
        print('Step %d: loss = %.2f' % (step, loss_value))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--batch_size',
    type=int,
    default=20,
    help='Batch Size'
  )
  parser.add_argument(
    '--log_dir',
    type=str,
    default='/tmp/tensorflow/quick-a-net',
    help='Directory to put the log data'
  )
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='Initial Learning Rate'
  )
  parser.add_argument(
    '--max_steps',
    type=int,
    default=2000,
    help='Number of steps to run trainer.'
  )
  logging.basicConfig(level=logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)