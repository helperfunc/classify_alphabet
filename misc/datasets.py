import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np
import tensorflow as tf
#from infogan.misc.utils import mkdir_p
from skimage.io import imsave

class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self):
    	"""
        data_directory = "MNIST"
         
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        #dataset = mnist.input_data.read_data_sets(data_directory, validation_size=35000)
        # only leave 25000 images for training
        dataset = mnist.input_data.read_data_sets(data_directory, validation_size=54500)
        self.train = dataset.train
        # self.train: DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state() ##????
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]]) # 10 samples
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels), ## labels
        )
        self.test = dataset.test
        self.validation = dataset.validation
        """
        self.tfrecord_filename = "/Users/huixu/Documents/codelabs/alphabet2cla/data_resized_TFRecord/train-00000-of-00001"
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)
        self.image_shape_size = (28, 28)
        #self.gen_img_dataset = self.get_gen_image(tfrecord_filename)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        #return 255.0-data
        #return 1.-data # save white pic
        return data # save black pic
    """
    def save_subset_as_images(self):
    	# save subset images in the training data as 5500 training dataset
    	batch_size = 55
    	data_size = 5500
    	epoch = data_size / batch_size
    	sub_imgs_dir = "5500sub"
    	mkdir_p(sub_imgs_dir)
    	for i in range(10):
            mkdir_p(os.path.join(sub_imgs_dir,str(i)))
        for e in range(epoch):
            batch_images_x, batch_images_y = self.train.next_batch(batch_size, shuffle=False)
            for img, real_label in zip(batch_images_x, batch_images_y):
				class_dir = os.path.join(sub_imgs_dir, str(real_label))
				already_num_pics_in_dir = len(os.listdir(class_dir))
				imsave(os.path.join(class_dir,str(already_num_pics_in_dir)+'.png'), img.reshape(self.image_shape_size))
	"""			
    def show_mnist(self, x, batch_size, generate_img_per_class_batch):
        stacked_img = []
        rows = 10
        cols = generate_img_per_class_batch
        imgs = tf.reshape(x, [rows, cols] + list(self.image_shape_size))
        #print(imgs.shape) #(10, 30, 28, 28)
        #exit(0)
        for row in xrange(rows):
            row_img = []
            for col in xrange(cols):
                row_img.append(imgs[row, col, :, :])
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.concat(0, stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        return imgs

    # Function to tell TensorFlow how to read a single image from input file
    def get_gen_image(self, filename):
      # convert filenames to a queue for an input pipeline.
      filenameQ = tf.train.string_input_producer([filename],num_epochs=None)
     
      # object to read records
      recordReader = tf.TFRecordReader()
    
      # read the full set of features for a single example 
      key, fullExample = recordReader.read(filenameQ)
    
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
      return (image,label)
      
    def gen_img_next_batch(self, gen_dataset, batch_size_num):
      # associate the "label" and "image" objects with the corresponding features read from 
      # a single example in the training data file
      #label, image = get_gen_image(tfrecord_filename)
      # associate the "label_batch" and "image_batch" objects with a randomly selected batch---
      # of labels and images respectively
      # https://github.com/tensorflow/models/blob/8505222ea1f26692df05e65e35824c6c71929bb5/differential_privacy/dp_sgd/dp_mnist/dp_mnist.py#L157
      imageBatch, labelBatch = tf.train.shuffle_batch([gen_dataset[0], gen_dataset[1]], batch_size=batch_size_num, capacity=1000 + 3 * batch_size_num, min_after_dequeue=1000)
      return imageBatch, labelBatch