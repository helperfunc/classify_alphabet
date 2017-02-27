#!/usr/bin/env python

from PIL import Image
import os, sys

def resizeImage(infile_dir, infile, output_dir="", size=(28,28)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]
     #print(infile)
     #print (outfile)
     #exit(0)
     if (cmp(extension, ".png")):
        return

     if infile != outfile:
        try :
            # im = Image.open(infile)
            im = Image.open(os.path.join(infile_dir,infile))
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(output_dir+outfile+extension,"PNG")
        except IOError:
            print "cannot reduce image for ", infile


if __name__=="__main__":
    for subdir in os.listdir('/Users/huixu/Documents/codelabs/alphabet2cla/data'):
        output_dir = "/Users/huixu/Documents/codelabs/alphabet2cla/data_resized/"+subdir+"/"
        #dire = os.getcwd()
        dire = "/Users/huixu/Documents/codelabs/alphabet2cla/data/"+subdir+"/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
        for f in os.listdir(dire):
            resizeImage(dire, f,output_dir)
        #exit(0)