import numpy as np
import tensorflow as tf
from skimage.io import imsave
import functools
import os
from tensorflow.python.layers import utils

class MnistDataset(object):
    def __init__(self, batch_size_num):
        self.train_tfrecord = "/Users/huixu/Documents/codelabs/alphabet2cla/data_resized_TFRecord/train-00000-of-00001"
        self.embedding_tfrecord = "/Users/huixu/Documents/codelabs/alphabet2cla/data_resized_TFRecord/train-00000-of-00001"
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)
        self.image_shape_size = (28, 28)
        self.batch_size_num = batch_size_num
        self.embedding_imgs = None
        self.embedding_labels = None

    #@define_scope
    # Function to tell TensorFlow how to read a single image from input file
    def get_an_image(self, tfrecord_file):
      # convert filenames to a queue for an input pipeline.
      queue = tf.train.string_input_producer([tfrecord_file],num_epochs=None)
      #embedding_queue = tf.train.string_input_producer([self.embedding_tfrecord],num_epochs=None)

      #queue = tf.QueueBase.from_list(self.data_type, [train_queue, embedding_queue])

      # object to read records
      recordReader = tf.TFRecordReader()

      # read the full set of features for a single example
      key, fullExample = recordReader.read(queue)

      # parse the full example into its' component features.
      features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
          })


      # now we are going to manipulate the label and image features

      label = features['image/class/label']
      image_buffer = features['image/encoded']

      # Decode the jpeg
      with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=0)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


      # cast image into a single array, where each element corresponds to the greyscale
      # value of a single pixel.
      # the "1-.." part inverts the image, so that the background is black.

      #image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[self.image_dim]) # if pic is white
      image=tf.reshape(tf.image.rgb_to_grayscale(image),[self.image_dim]) # if pic is black
      #image = tf.decode_raw(features['image/encoded'], tf.uint8)
      #image.set_shape([self.image_dim])
      #http://stackoverflow.com/questions/37981536/tensorflow-python-framework-errors-outofrangeerror
      # re-define label as a "one-hot" vector
      # it will be [0,1] or [1,0] here.
      # This approach can easily be extended to more classes.
      #label=tf.pack(tf.one_hot(label-1, nClass))
      label = label-1
      """
      if self.data_type == 0:
          batch_imgs, batch_labels = tf.train.shuffle_batch([image, label], batch_size=self.batch_size_num, capacity=1000 + 3 * self.batch_size_num, min_after_dequeue=1000)
      else:
          batch_imgs, batch_labels = tf.train.batch([image, label], 1024, capacity=1024)
          if not os.path.exists("/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/sprite_1024.png"):
              self.generate_sprite(batch_imgs, batch_labels)
      return batch_imgs,batch_labels
      """
      return image, label

    #@define_scope
    def get_train_batch_images(self):
        batch_imgs, batch_labels = tf.train.shuffle_batch(self.get_an_image(self.train_tfrecord), batch_size=self.batch_size_num, capacity=1000 + 3 * self.batch_size_num, min_after_dequeue=1000)
        return batch_imgs, batch_labels

    def get_eval_batch_images(self):
        batch_imgs, batch_labels = tf.train.shuffle_batch(self.get_an_image(self.embedding_tfrecord), batch_size=self.batch_size_num, capacity=1000 + 3 * self.batch_size_num, min_after_dequeue=1000)
        return batch_imgs, batch_labels

    #@define_scope
    def get_embedding_images(self):
        batch_imgs, batch_labels = tf.train.batch(self.get_an_image(self.embedding_tfrecord), batch_size=self.batch_size_num, capacity=self.batch_size_num)
        #if self.embedding_labels == None:
        self.embedding_imgs, self.embedding_labels = self.generate_sprite(batch_imgs, batch_labels)
        #return batch_imgs, batch_labels
        #print(self.embedding_labels)

    def generate_sprite(self, sprite, sprite_labels):
        #dataset = MnistDataset()
        #imgs, labels = dataset.get_gen_image(dataset.tfrecord_filename)
        #sprite, sprite_labels = dataset.gen_img_next_batch(dataset.get_gen_image(dataset.tfrecord_filename), 1024)
        #sprite, sprite_labels = self.get_gen_image(1, 1024)
        coord = tf.train.Coordinator()

        with tf.Session() as sess:
            # new session is ok?
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            sprite_data, sprite_labels_data = sess.run([sprite, sprite_labels])
            if not os.path.exists("/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/sprite_1024.png"):
                imgs = pile_up(sprite_data, 26, 26, self.image_shape_size)
                imgs = np.asarray(imgs.eval()).reshape((28*(676/26),28*26))

                imsave('/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/sprite_1024.png', imgs)
                labels_tsv(sprite_labels_data)
            coord.request_stop()
            coord.join(threads)
        #return tf.constant(sprite_data), tf.constant(sprite_labels_data)
        return sprite_data, sprite_labels_data

    #def input_switch(self, is_train_data=True):
        """
        input switch between training TFRecords queue or validation TFRecords queue
        """
    #    return utils.smart_cond(is_train_data, self.get_train_batch_images, self.get_valid_batch_images)

def pile_up(x, rows, cols, image_shape_size):
    """save sprite images"""
    stacked_img = []
    imgs = tf.reshape(x, [rows, cols] + list(image_shape_size))
    #print(imgs.shape) #(10, 30, 28, 28)
    #exit(0)
    for row in xrange(rows):
        row_img = []
        for col in xrange(cols):
            row_img.append(imgs[row, col, :, :])
        stacked_img.append(tf.concat(axis=1, values=row_img))
    imgs = tf.concat(axis=0, values=stacked_img)
    imgs = tf.expand_dims(imgs, 0)
    #imsave('/Users/huixu/Documents/codelabs/alphabet2cla/logs/sprite_1024.png', imgs)
    return imgs

def labels_tsv(sprite_labels):
    """save labels as .tsv file"""
    labels_file = '/Users/huixu/Documents/codelabs/alphabet2cla/misc/labels.txt'
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    #print(unique_labels) #['p', 'q']
    #exit(0)
    with open('/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/labels_1024.tsv', 'w') as f:
        for label in sprite_labels:
            #print(label) # 1 should be turned to q
            #exit(0)

            f.write(unique_labels[label])
            f.write('\n')
