import tensorflow as tf
import numpy as np

class SketchANet(object):
  """Define a Sketch-A-Net Model"""

  def __init__(self,
        input_shape=(15, 15, 6, 64),
        init_stddev=0.1,
        dropout_prob=0.5,
        learning_rate=0.001,
        decay_steps=100, decay_rate=0.96, staircase=True):
    self._input_shape = input_shape
    self._init_stddev = stddev

    self._dropout_prob = dropout_prob
    self._learning_rate = learning_rate
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    self._staircase = staircase


  def inference(self,images, training=True):
    """This prepares the tensorflow graph for the vanilla Sketch-A-Net network
    and returns the tensorflow Op from the last fully connected layer

    Args:
      images: the input images of shape (N, H, W, C) for the network returned from the data layer
    
    Returns:
      Logits for the softmax loss
    """
    keep_prob = self._dropout_prob if training else 1.0

    # Layer 1
    with tf.name_scope('L1') as scope:
      weights1 = tf.Variable(tf.truncated_normal(
          (15, 15, 6, 64), stddev=self._init_stddev), name='weights')
      biases1 = tf.Variable(tf.constant(self._init_stddev, shape=(64,)), name='biases')
      
      conv1 = tf.nn.conv2d(images, weights1, [1, 3, 3, 1], padding='VALID', name='conv')
      relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name='relu')
      pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
          padding='VALID', name='pool')
    
    # Layer 2
    with tf.name_scope('L2') as scope:
      weights2 = tf.Variable(tf.truncated_normal(
          (5, 5, 64, 128), stddev=self._init_stddev), name='weights')
      biases2 = tf.Variable(tf.constant(self._init_stddev, shape=(128,)), name='biases')
      
      conv2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1], padding='VALID', name='conv')
      relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name='relu')
      pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
          padding='VALID', name='pool')

    # Layer 3
    with tf.name_scope('L3') as scope:
      weights3 = tf.Variable(tf.truncated_normal(
          (3, 3, 128, 256), stddev=self._init_stddev), name='weights')
      biases3 = tf.Variable(tf.constant(self._init_stddev, shape=(256,)), name='biases')
      
      conv3 = tf.nn.conv2d(pool2, weights3, [1, 1, 1, 1], padding='SAME', name='conv')
      relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3), name='relu')

    # Layer 4
    with tf.name_scope('L4') as scope:
      weights4 = tf.Variable(tf.truncated_normal(
          (3, 3, 256, 256), stddev=self._init_stddev), name='weights')
      biases4 = tf.Variable(tf.constant(self._init_stddev, shape=(256,)), name='biases')
      
      conv4 = tf.nn.conv2d(pool3, weights4, [1, 1, 1, 1], padding='SAME', name='conv')
      relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4), name='relu')

    # Layer 5
    with tf.name_scope('L5') as scope:
      weights5 = tf.Variable(tf.truncated_normal(
          (3, 3, 256, 256), stddev=self._init_stddev), name='weights')
      biases5 = tf.Variable(tf.constant(self._init_stddev, shape=(256,)), name='biases')
      
      conv5 = tf.nn.conv2d(pool4, weights5, [1, 1, 1, 1], padding='SAME', name='conv')
      relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5), name='relu')
      pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
          padding='VALID', name='pool')

    # Layer 6
    with tf.name_scope('L6') as scope:
      weights6 = tf.Variable(tf.truncated_normal(
          (7, 7, 512, 512), stddev=self._init_stddev), name='weights')
      biases6 = tf.Variable(tf.constant(self._init_stddev, shape=(512,)), name='biases')
      
      fc6 = tf.nn.conv2d(pool5, weights6, [1, 1, 1, 1], padding='SAME', name='conv')
      relu6 = tf.nn.relu(tf.nn.bias_add(conv6, biases6), name='relu')
      dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob, name='dropout')

    # Layer 7
    with tf.name_scope('L7') as scope:
      weights7 = tf.Variable(tf.truncated_normal(
          (1, 1, 512, 512), stddev=self._init_stddev), name='weights')
      biases7 = tf.Variable(tf.constant(self._init_stddev, shape=(512,)), name='biases')
      
      fc7 = tf.nn.conv2d(dropout6, weights7, [1, 1, 1, 1], padding='SAME', name='conv')
      relu7 = tf.nn.relu(tf.nn.bias_add(conv7, biases7), name='relu')
      dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob, name='dropout')
  
    # Layer 8
    with tf.name_scope('L8') as scope:
      weights8 = tf.Variable(tf.truncated_normal(
          (1, 1, 512, 250), stddev=self._init_stddev), name='weights')
      biases8 = tf.Variable(tf.constant(self._init_stddev, shape=(250,)), name='biases')
      
      fc8 = tf.nn.conv2d(dropout7, weights8, [1, 1, 1, 1], padding='SAME', name='conv')
      fc8b = tf.nn.bias_add(fc8, biases8)
    
    logits = tf.reshape(fc8b, [-1, 250])

    return logits

  def loss(self, logits, labels):
    """Applies the softmax loss to given logits

    Args:
      logits: the logits obtained from the inference graph
      labels: the ground truth labels for the respective images

    Returns:
      The loss value obtained form the softmax loss applied
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    xentropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('loss', xentropy_mean)
    return xentropy_mean

  def training(loss):
    """Returns the training Op for the loss function using the AdamOptimizer

    Args:
      loss: the loss calculated by the loss function

    Returns:
      train_op: the tensorflow's trainig Op
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
    train_op = optimizer.minimize(loss)

    tf.summary.scalar('global step', global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    return train_op

  def evaluation(logits, labels, k=1, training=True):
    """Evaluates the number of correct predictions for the given logits and labels
    Args:
      logits: the logits obtained from the inference graph
      labels: the ground truth labels
      k: correct in top-k guesses
      training: if used at training time with the batch
    
    Return:
        Returns the number of correct predictions
    """
    if not training:
        logits = tf.reduce_sum(tf.reshape(logits, [10, -1, 250]), axis=0)
    correct = tf.nn.in_top_k(logits, tf.cast(labels[:tf.shape(logits)[0]], tf.int32), k)
    return tf.reduce_sum(tf.cast(correct, tf.int32))