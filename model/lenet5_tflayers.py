from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import learn
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from misc.save_sprite_labels import generate_sprite
from misc.datasets import MnistDataset
from progressbar import ETA, Bar, Percentage, ProgressBar

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string(
    'save_path', '/Users/huixu/Documents/codelabs/alphabet2cla/logs/', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('log_dir', '/Users/huixu/Documents/codelabs/alphabet2cla/logs/',
                           'Where to save the logs for visualization in TensorBoard.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 666
EPOCH_SIZE = 52832 // BATCH_SIZE

# Our application logic will be added here

def conv_layer(input, size_in, size_out, training, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        #nor_conv = normalization_layers.BatchNormalization().apply(conv, training=training)
        act = tf.nn.relu(conv + b)
        #act = tf.nn.relu(nor_conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
def fc_layer(input, size_in, size_out, training, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        #act = tf.nn.relu( normalization_layers.BatchNormalization().apply( tf.matmul(input, w) + b,  training=training) )
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activations", act)
        return act
        
def lenet5_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()
    
    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    #y = tf.placeholder(tf.float32, shape=[None, 52], name="labels")
    y = tf.placeholder(tf.int64, shape=[None], name="labels")
    # tf.nn.sparse_softmax_cross_entropy_with_logits can't take one hot labels
    training = array_ops.placeholder(dtype='bool') # for batch noramlization and dropout
    
    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 20, training, "conv1")
        conv_out = conv_layer(conv1, 20, 50, training,"conv2")
    else:
        conv1 = conv_layer(x_image, 1, 50, training, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 50])
    
    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 50, 500, training, "fc1")
        embedding_input = fc1
        embedding_size = 500
        
        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(inputs=fc1, rate=0.4, training=training, name="dropout")
  
        #logits = fc_layer(fc1, 1024, 52, training, "fc2")
        logits = fc_layer(dropout, 500, 52, training, "fc2")
    else:
        embedding_input = flattened
        embedding_size = 7 * 7 * 50
        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(inputs=logits, rate=0.4, training=training, name="dropout")
        #logits = fc_layer(flattened, 7 * 7 * 50, 52, training, "fc")
        logits = fc_layer(dropout, 7 * 7 * 50, 52, training, "fc")
        
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)
        
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        
    with tf.name_scope("accuracy"):
        #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        correct_prediction = tf.equal(tf.argmax(logits, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }
    
    summ = tf.summary.merge_all()
    
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    # first 1024 is 1024 images, second is daoshu di er ceng neurons
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.log_dir + hparam)
    writer.add_graph(sess.graph)
    #writer.close()
    #exit(0)
    
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    
    dataset = MnistDataset()
    embedding_imgs, embedding_labels = generate_sprite(dataset)
    # Convert to 1-hot representation.
    #embedding_labels = (np.arange(52) == embedding_labels[:, None]).astype(np.float32)
    
    embedding_config.sprite.image_path = FLAGS.log_dir + 'sprite_1024.png'
    embedding_config.metadata_path = FLAGS.log_dir + 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    #for epoch in range(20):
    # http://stackoverflow.com/questions/41276012/sess-runtensor-does-nothing
    batch_xs, batch_ys = dataset.gen_img_next_batch(dataset.get_gen_image(dataset.tfrecord_filename), BATCH_SIZE)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    widgets = ["epoch #%d|" % 1, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=20001, widgets=widgets)
    pbar.start()
    for i in range(20001):
    	pbar.update(i)
        train_images, train_labels = sess.run([batch_xs, batch_ys]) # is it fetch different data everytime? yes
        #print('train_label 0: %d' % train_labels[0])
        #print(train_images[0])
        #print(train_labels[np.argmax(train_labels)]) # 51
        #exit(0)
        # Convert to dense 1-hot representation.
        #train_labels = (np.arange(52) == train_labels[:, None]).astype(np.float32)
        
        #batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            #[train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: train_images, y: train_labels, training:False})
            writer.add_summary(s, i)
            print(i,'-',train_accuracy)
        if i % 500 == 0:
            #sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y:minst.test.labels[:1024]})
            sess.run(assignment, feed_dict={x: embedding_imgs, y:embedding_labels, training:False})
            # how to make sure they are the same as sprite images?
            
            embedding_imgs_accuracy = sess.run(accuracy, feed_dict={x: embedding_imgs, y:embedding_labels, training:False})
            print(embedding_imgs_accuracy)
            
            saver.save(sess, os.path.join(FLAGS.save_path, "model.ckpt"), i)
        #sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
        sess.run(train_step, feed_dict={x: train_images, y: train_labels, training:True})
    
    coord.request_stop()
    coord.join(threads)
    writer.close()
    sess.close()
    
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main(_=None):
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:
        
        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (e.g. "lr_1E-3,fc=2,conv=2")
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)
                
                # Actually run with the new settings
                lenet5_model(learning_rate, use_two_fc, use_two_conv, hparam)
                
                
if __name__ == "__main__":
    tf.app.run()