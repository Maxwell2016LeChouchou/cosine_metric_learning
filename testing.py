import os
import numpy as np
import cv2
import csv

def _parse_filename(csv_dir):

    csv_dir = '/home/maxwell/Downloads/cosine_metric_learning/datasets/test_csv'
    for filename in os.listdir(csv_dir):
        for line in open(os.path.join(csv_dir,filename), "r"):
            data = line.split(",")
            filename = data[0]
            filename_base, ext = os.path.splitext(filename)
            person_name, dir_file, frame_idx = filename_base.split("/")
            print(person_name)
            print(dir_file)
            print(frame_idx)

    return str(filename), str(person_name), str(dir_file), str(frame_idx)



def train_dataset(image_path):
    
    image_list = []
    yt_person_name = []
    yt_dir_file = []

    

    print(image_list)
    print(yt_person_name)
    print(yt_dir_file) 


def main():
    input_path = '/home/maxwell/Downloads/cosine_metric_learning/datasets/test_dataset/'
    csv_dir = '/home/maxwell/Downloads/cosine_metric_learning/datasets/test_csv'
    train_dataset(input_path)


if __name__ == "__main__":
    main()






import tensorflow as tf
 2 import numpy as np
 3 import glob
 4 import os
 5 from PIL import Image
 6 def _int64_feature(value):
 7     if not isinstance(value,list):
 8         value=[value]
 9     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
10 def _byte_feature(value):
11     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
12 def encode_to_tfrecords(data_path,name,rows=24,cols=16):#从图片路径读取图片编码成tfrecord
13     folders=os.listdir(data_path)#这里都和我图片的位置有关
14     folders.sort()
15     numclass=len(folders)
16     i=0
17     npic=0
18     writer=tf.python_io.TFRecordWriter(name)
19     for floder in folders:
20         path=data_path+"/"+floder
21         img_names=glob.glob(os.path.join(path,"*.bmp"))
22         for img_name in img_names:
23             img_path=img_name
24             img=Image.open(img_path).convert('P')
25             img=img.resize((cols,rows))
26             img_raw=img.tobytes()
27             labels=[0]*34#我用的是softmax，要和预测值的维度一致
28             labels[i]=1
29             example=tf.train.Example(features=tf.train.Features(feature={#填充example
30                 'image_raw':_byte_feature(img_raw),
31                 'label':_int64_feature(labels)}))
32             writer.write(example.SerializeToString())#把example加入到writer里，最后写到磁盘。
33             npic=npic+1
34         i=i+1
35     writer.close()
36     print npic
37 
38 def decode_from_tfrecord(filequeuelist,rows=24,cols=16):
39     reader=tf.TFRecordReader()#文件读取
40     _,example=reader.read(filequeuelist)
41     features=tf.parse_single_example(example,features={'image_raw':#解码
42             tf.FixedLenFeature([],tf.string),
43             'label':tf.FixedLenFeature([34,1],tf.int64)})
44     image=tf.decode_raw(features['image_raw'],tf.uint8)
45     image.set_shape(rows*cols)
46     image=tf.cast(image,tf.float32)*(1./255)-0.5
47     label=tf.cast(features['label'],tf.int32)
48     return image,label
49     
50 
51 def get_batch(filename_queue,batch_size):
52     with tf.name_scope('get_batch'):
53         [image,label]=decode_from_tfrecord(filename_queue)
54         images,labels=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=2,
55                  capacity=100+3*batch_size,min_after_dequeue=100)  
56         return images,labels
57 
58     
59 def generate_filenamequeue(filequeuelist):
60     filename_queue=tf.train.string_input_producer(filequeuelist,num_epochs=5)
61     return filename_queue
62 
63 def test(filename,batch_size):
64     filename_queue=generate_filenamequeue(filename)
65     [images,labels]=get_batch(filename_queue,batch_size)
66     init_op = tf.group(tf.global_variables_initializer(),
67                        tf.local_variables_initializer())
68     sess=tf.InteractiveSession()
69     sess.run(init_op)
70     coord=tf.train.Coordinator()
71     threads=tf.train.start_queue_runners(sess=sess,coord=coord)
72     i=0
73     try:
74         while not coord.should_stop():
75             image,label=sess.run([images,labels])
76             i=i+1
77             if i%1000==0:
78                 for j in range(batch_size):#之前tfrecord编码的时候，数据范围变成[-0.5,0.5],现在相当于逆操作，把数据变成图片像素值
79                     image[j]=(image[j]+0.5)*255
80                     ar=np.asarray(image[j],np.uint8)
81                     #image[j]=tf.cast(image[j],tf.uint8)
82                     print ar.shape
83                     img=Image.frombytes("P",(16,24),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
84                     img.save("/home/wen/MNIST_data/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看
85             '''if(i>710):
86                 print("step %d"%(i))
87                 print image
88                 print label'''
89     except tf.errors.OutOfRangeError:
90         print('Done training -- epoch limit reached')
91     finally:
92     # When done, ask the threads to stop.
93         coord.request_stop()
94 
95 # Wait for threads to finish.
96     coord.join(threads)
97     sess.close()