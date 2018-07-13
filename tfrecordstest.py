# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import scipy.misc as misc
import numpy as np


imgdir='images/nobeard/'

lstimg=os.listdir(imgdir)


'''make tfrecords '''

#writer=tf.python_io.TFRecordWriter("tfrecords/nobeard.tfrecords")
##writer=tf.python_io.TFRe
#cout=0
#for imgname in lstimg:
#    print(cout)
#    cout+=1
#    img_path=imgdir+imgname
#    img=misc.imread(img_path)
#    img_raw=img.tobytes()
#    example=tf.train.Example(features=tf.train.Features(feature={
#            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
##    example = tf.train.Example(features=tf.train.Features(feature={
##            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
##            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
##        }))
#
#    writer.write(example.SerializeToString())
#    
#writer.close()

''' '''

#read tfrecords

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [160, 160, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#    label = tf.cast(features['label'], tf.int32)

    return img#, label

img=read_and_decode("test.tfrecords")

imgbatch=tf.train.shuffle_batch([img],batch_size=1,capacity=200,min_after_dequeue=50)
#imgbatch=tf.train.batch([img],batch_size=1)

init = tf.initialize_all_variables()

epochs=2

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(epochs):
#        imgbatch=tf.train.shuffle_batch([img],batch_size=1,capacity=20,min_after_dequeue=10)
#        imgbatch=tf.train.batch([img],batch_size=1,capacity=20)
       
        for j in range(200):
            val= sess.run([imgbatch])
            val=np.array(val[0])
            misc.imsave('{}/{}test.png'.format(i,j),val[0])
            print(val[0].shape)
            
    coord.request_stop()  
    coord.join(threads)

