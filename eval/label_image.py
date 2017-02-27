# https://github.com/llSourcell/tensorflow_image_classifier/blob/master/src/label_image.py

import tensorflow as tf
import os, sys
import PIL
from PIL import Image
import numpy as np

def rescale_image(image_path):
    size = 28, 28
    
    #for infile in image_path:
    infile = image_path
    outfile = os.path.splitext(infile)[0] + "_re.png"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "PNG")
        except IOError:
            print "cannot create thumbnail for '%s'" % infile
    return outfile

# change this as you see fit
image_path = sys.argv[1]
image_path = rescale_image(image_path)
#print(os.path.join('/Users/huixu/Documents/codelabs/alphabet2cla',image_path))
#exit(0)
# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# https://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
image = Image.open(image_path).convert('LA')

image_data = np.array(image)[:, :, 0].reshape(1, 784)[0] / 255.0
#print(image_data/255.0)
#exit(0)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/Users/huixu/Documents/codelabs/alphabet2cla/misc/labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/Users/huixu/Documents/codelabs/alphabet2cla/logs/frozen_model.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    


with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')
    
    predictions = sess.run(softmax_tensor, \
             {'x:0':image_data, 'Placeholder:0': False})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))