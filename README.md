# 分类字母

本代码库包含了用TensorFlow 1.0分类英文字母的代码。包括：1）导入文件夹中的图片数据；2）分类模型的构建；
3）模型图与分类结果的可视化；4）保存freez后的常数模型；5）测试单张图片的分类结果；这一整套深度学习模型构建和评价过程。
该过程只是能起作用，并非代表了最优流程。
比如图片数据保存在了TFRecords中，但导入到图中用到了feed_dict。
再如freez模型，即常数化各节点，一般是用于嵌入式模型中。
还比如代码没有包括Android等移动端模型的部署代码。
代码大多属于拼凑而得，然觉其可能对和我一样的个人学习者有帮助，遂开源。

## 1)导入数据

训练数据集下载路径：http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ 
选用的数据集是其中的EnglishFnt.tgz，原数据集为128*128的灰度图像。

### 1.1 生成28*28的灰度图像
新建一个文件夹data，将下载的图片Fnt/Sample011 ~ Fnt/Sample062拷贝到当前新建文件夹data下，目录结构为:
|-data
|--Sample011
...
|--Sample062
修改misc/resize_image.py中的data和data_resized文件夹路径，执行：
    $ python misc/resize_image.py
 
得到28*28的灰度图像。data_resized所在的目录结构为：
|-data_resized
|--Sample011
...
|--Sample062

### 1.2 生成TFRecords
修改misc/gen_labels.py中的路径，根据data_resized的子文件夹名生成labels.txt文件。
    $ python misc/gen_labels.py 
新建一个data_resized_TFRecord文件夹，修改misc/build_image_data.py中的train_directory，output_directory和labels_file的路径。生成TFRecord。
    $ python misc/build_image_data.py
    
备注：如果图片是RGB可以直接用[models中的示例](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py)。

修改misc/dataset.py中self.tfrecord_filename的路径，指向build_image_data.py生成的TFRecords。

### 1.3 从TFRecords中随机读取batch size的图片数据
需要设置一个queue runner thread来将图片数据加入到输入数据队列中，见[sess.run(Tensor) does nothing](https://stackoverflow.com/questions/41276012/sess-runtensor-does-nothing)。

## 2. 构建模型
目前实现了卷积层和全连接层，以及训练过程中全连接层的dropout。待实现：batch normalization。

## 3. 可视化

    writer = tf.summary.FileWriter(FLAGS.log_dir + hparam)
    writer.add_graph(sess.graph)
    
将Tensor流动的图存入logs文件中，可以用TensorBoard可视化。可以改变这条语句的位置，这样保存的就变成了一部分Tensors。
将最后一层的输入层作为embedding的向量。增加了[sprite image和对应labels](https://www.tensorflow.org/get_started/embedding_viz#images)的生成代码。
具体见model/lenet5_tflayers.py。

## 4. Freez图
将图结构的定义和各权值合并在一起得到训练后的模型。若在lenet5_tflayers.py中修改了模型的结构，需要在eval/save_pb.py中修改`output_node_names`，可以用TensorBoard可视化图选项卡查看类别正确率对应的节点。
![TensorBoard模型图](https://cloud.githubusercontent.com/assets/19688861/23507401/e21fdcac-ff87-11e6-968e-b042295b5a0b.png)

## 5. 测试单张图片的分类结果
修改eval/label_image.py中的label文件和pb文件的路径。如果在lenet5_tflayers.py中修改了模型的结构，需要修改文件中的softmax_tensor和predictions的feed_dict中的Tensor名。

执行`python eval/label_image.py 待分类的图片路径`，即可得到该图片的类别。

发现：目前数据集的数据全是正的图片，对于倾斜的字母图片，模型很难正确分类。
