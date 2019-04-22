import numpy as np
import tensorflow as tf
from collections import Counter
from string import punctuation

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
tf.reset_default_graph() 

sess = tf.Session('', tf.Graph())
with sess.graph.as_default():
    saver = tf.train.import_meta_graph('checkpoints/batmansplit.ckpt.meta')
    saver.restore(sess, "checkpoints/batmansplit.ckpt")
    # Restore variables from disk.
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval(session=sess))
    print("v2 : %s" % v2.eval())
