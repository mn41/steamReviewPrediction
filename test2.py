import numpy as np
import tensorflow as tf
from collections import Counter
from string import punctuation

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001

graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')



# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
tf.reset_default_graph() 
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('checkpoints/batmansplit.ckpt.meta')
    saver.restore(sess, "checkpoints/batmansplit.ckpt")
    # Restore variables from disk.
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval(session=sess))
    print("v2 : %s" % v2.eval())
