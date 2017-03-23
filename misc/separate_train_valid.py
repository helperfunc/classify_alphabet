"""
dataset_dir structure
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

 valid_dir structure
   data_dir/label_0/image0.jpeg
   data_dir/label_0/image1.jpg
   ...
   data_dir/label_0/image20.jpg
   ...
   data_dir/label_1/weird-image.jpeg
   data_dir/label_1/my-image.jpeg
   ...

 what left in dataset_dir is the training dataset
"""
import os
import shutil

dataset_dir = '/Users/huixu/Downloads/English/Fnt'
#train_dir = '/Users/huixu/Downloads/English/train'
valid_dir = '/Users/huixu/Downloads/English/valid'
valid_num_per_class = 20

def is_make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# move images in dataset_dir to valid_dir
# each class in valid_dir owns valid_num_per_class images
def mv_valid(dir_name):
    for sub_dir in os.listdir(dir_name):
        sub_dir_full = os.path.join(dir_name,sub_dir)
        valid_sub_dir = os.path.join(valid_dir,sub_dir)
        if not os.path.exists(valid_sub_dir):
            os.mkdir(valid_sub_dir)
        counter = 0
        for filename in os.listdir(sub_dir_full):
            if counter < valid_num_per_class:
                shutil.move(os.path.join(sub_dir_full, filename), valid_sub_dir)
                counter += 1
            else:
                break

def main():
    # build directory for training and validation dataset
    #is_make_dir(train_dir)
    is_make_dir(valid_dir)
    mv_valid(dataset_dir)

if __name__ == '__main__':
    main()
