# save sprite labels for embedding
# return the saved images for processing by model

import tensorflow as tf
import numpy as np
#from misc.datasets import MnistDataset
from skimage.io import imsave


def generate_sprite(dataset):
    #dataset = MnistDataset()
    #imgs, labels = dataset.get_gen_image(dataset.tfrecord_filename)
    #sprite, sprite_labels = dataset.gen_img_next_batch(dataset.get_gen_image(dataset.tfrecord_filename), 1024)
    sprite, sprite_labels = tf.train.batch(dataset.get_gen_image(dataset.tfrecord_filename), 1024, capacity=1024)
    coord = tf.train.Coordinator()
    
    with tf.Session() as sess:
    	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        sprite_data, sprite_labels_data = sess.run([sprite, sprite_labels])
        #print(sprite_labels_data)
        imgs = pile_up(sprite_data, 32, 32, dataset.image_shape_size)
        imgs = np.asarray(imgs.eval()).reshape((28*(1024/32),28*32))
        
        imsave('/Users/huixu/Documents/codelabs/alphabet2cla/logs_test/sprite_1024.png', imgs)
        labels_tsv(sprite_labels_data)
        coord.request_stop()
        coord.join(threads)
    return sprite_data, sprite_labels_data
        #exit(0)

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
			
#if __name__ == '__main__':
#    main()