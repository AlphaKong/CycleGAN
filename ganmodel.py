# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import scipy.misc as misc
from networklayers import build_generator_resnet_6blocks_d,build_gen_discriminator
from nextbatch import next_batch_dataset,ImagePool
from utils import scale_back,normalize_image,merge
import time
import datetime




class CycleGan(object):
    
    def __init__(self, epoch=5000,to_restore=False,mse_penalty=10.0,checkpiont_dir='checkpoint/',
                 lr=0.0003,batch_size=1,img_width=120,img_height=120,img_layer=3,pool_maxsize=50,
                 sample_steps=50,A_dir='tfrecords/nosmile.tfrecords',B_dir='tfrecords/smile.tfrecords',totalnum=2000):
        self.mse_penalty=mse_penalty
        self.to_restore=to_restore
        self.epoch=epoch
        self.lr=lr
        self.batch_size=batch_size
        self.img_width=img_width
        self.img_height=img_height
        self.img_layer=img_layer
        self.A_dir=A_dir
        self.B_dir=B_dir
        self.sample_steps=sample_steps
        self.checkpiont_dir=checkpiont_dir
        self.pool_maxsize=pool_maxsize
        self.pool=ImagePool(self.pool_maxsize)
        self.totalnum=totalnum
        self.model_name="Net.model"
        
        
    
    def checkpoint(self, saver, step, sess):
        model_name = self.model_name
        model_dir = self.checkpiont_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(sess, os.path.join(model_dir, model_name), global_step=step)
    
    def sample_model(self, rA,rB, epoch,count,sess):
        fake_B,fake_A = sess.run(
            [self.fake_B,self.fake_A],
            feed_dict={self.input_A: rA,self.input_B: rB}
        )
        
        a1=scale_back(rA)
        a2=scale_back(fake_B)
        
        b1=scale_back(rB)
        b2=scale_back(fake_A)
       
        merged_pair = np.concatenate([a1,a2,b1,b2], axis=2)
        merged_pair=merged_pair.reshape((merged_pair.shape[1],merged_pair.shape[2],merged_pair.shape[3]))

        s_dir='samples/'
        
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)
        
        sample_img_path = os.path.join(s_dir, "sample_%02d_%04d.png" % (epoch, count))
        misc.imsave(sample_img_path, merged_pair)
        

 

    def infer(self):
        imga=self.read_and_decode(self.A_dir)
        imgA_batch=tf.train.batch([imga],batch_size=1,capacity=self.totalnum)
        imgb=self.read_and_decode(self.B_dir)
        imgB_batch=tf.train.batch([imgb],batch_size=1,capacity=self.totalnum)
        self.buildmodel()
        
        result_dir="images/results/smiles/"
        checkpiont_dir="checkpoint/smiles/"
        
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())  
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            
            ckpt = tf.train.latest_checkpoint(checkpiont_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print("restored model %s" % checkpiont_dir)
            else:
                print("fail to restore model %s" % checkpiont_dir)
                return
            
            
            if not os.path.exists(result_dir):
                os.makedirs(result_dir) 
            
            nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            start_time=nowTime
            for i in range(0,self.totalnum):
                rA,rB=sess.run([imgA_batch,imgB_batch])#  
                    
                rA=normalize_image(np.array(rA))
#                    print(rA.shape)
                rB=normalize_image(np.array(rB))

                fake_B,fake_A = sess.run(
                    [self.fake_B,self.fake_A],
                    feed_dict={self.input_A: rA,self.input_B: rB}
                )
                
                a1=scale_back(rA)
                a2=scale_back(fake_B)
                
                b1=scale_back(rB)
                b2=scale_back(fake_A)
               
                merged_pair = np.concatenate([a1,a2,b1,b2], axis=2)
                merged_pair=merged_pair.reshape((merged_pair.shape[1],merged_pair.shape[2],merged_pair.shape[3]))

                
                sample_img_path = os.path.join(result_dir, "sample_%04d.png" % (i))
                misc.imsave(sample_img_path, merged_pair)
                print(sample_img_path)
                
            end_Time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('start_time: {0}, and end_time: {1}'.format(start_time,end_Time))  
            coord.request_stop()  
            coord.join(threads)
    
    
    def read_and_decode(self,filename):
        #根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([filename])
    
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image' : tf.FixedLenFeature([], tf.string),
                                           })
    
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [self.img_height, self.img_width, self.img_layer])
 
        return img
      
    def init_data(self):
#        with open(self.A_dir,'rb') as f1:
#            Adata=np.load(f1)
#        
#        with open(self.B_dir,'rb') as f2:
#            Bdata=np.load(f2)
#        self.totalnum=Adata.shape[0]
#        self.traindata=next_batch_dataset(Adata,Bdata,self.totalnum)
       pass
        
       
    
    def buildmodel(self):
        
        self.input_A = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_layer], name="input_B")
     
        #输入 real A
        self.fake_B= build_generator_resnet_6blocks_d(self.input_A,reuse=False,name="g_A2B")
        self.cyc_A=build_generator_resnet_6blocks_d(self.fake_B,reuse=False,name="g_B2A")
        #输入 real B
        self.fake_A= build_generator_resnet_6blocks_d(self.input_B,reuse=True,name="g_B2A")
        self.cyc_B=build_generator_resnet_6blocks_d(self.fake_A,reuse=True,name="g_A2B")
        
        self.DB_fake=build_gen_discriminator(self.fake_B,reuse=False,name="d_B")
        self.DA_fake=build_gen_discriminator(self.fake_A,reuse=False,name="d_A")
        
     
       #L1
#        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        self.cyc_loss=self.mse_penalty *tf.reduce_mean(tf.abs(self.input_B-self.cyc_B)) \
                    +self.mse_penalty *tf.reduce_mean(tf.abs(self.input_A-self.cyc_A))
#        
#        self.cyc_loss=self.mse_penalty *tf.reduce_mean(tf.squared_difference(self.input_B,self.cyc_B)) \
#                    +self.mse_penalty *tf.reduce_mean(tf.squared_difference(self.input_A,self.cyc_A))
# 
        
#        #mse
#        self.cyc_loss=self.mse_penalty *tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.input_B, self.cyc_B), [1, 2, 3]))\
#                    +self.mse_penalty *tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.input_A, self.cyc_A), [1, 2, 3]))
#      
        self.g_loss_a2b = tf.reduce_mean(tf.squared_difference(self.DB_fake,tf.ones_like(self.DB_fake)))+self.cyc_loss
        self.g_loss_b2a = tf.reduce_mean(tf.squared_difference(self.DA_fake,tf.ones_like(self.DA_fake)))+self.cyc_loss

        self.g_loss=tf.reduce_mean(tf.squared_difference(self.DB_fake,tf.ones_like(self.DB_fake))) \
                            +tf.reduce_mean(tf.squared_difference(self.DA_fake,tf.ones_like(self.DA_fake))) \
                            +self.cyc_loss
        
                
        self.fake_A_sample = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_layer],  name="fake_A_sample")
        self.fake_B_sample = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_layer],  name="fake_B_sample")
   
        self.DA_real=build_gen_discriminator(self.input_A,reuse=True,name="d_A")
        self.DB_real=build_gen_discriminator(self.input_B,reuse=True,name="d_B")
        
        self.DA_fake_sample = build_gen_discriminator(self.fake_A_sample,reuse=True,name="d_A")
        self.DB_fake_sample = build_gen_discriminator(self.fake_B_sample,reuse=True,name="d_B")

        self.db_loss_real=tf.reduce_mean(tf.squared_difference(self.DB_real,tf.ones_like(self.DB_real)))
        self.db_loss_fake=tf.reduce_mean(tf.squared_difference(self.DB_fake_sample,tf.zeros_like(self.DB_fake_sample)))
        self.db_loss=0.5*(self.db_loss_real+self.db_loss_fake)
        
        self.da_loss_real=tf.reduce_mean(tf.squared_difference(self.DA_real,tf.ones_like(self.DA_real)))
        self.da_loss_fake=tf.reduce_mean(tf.squared_difference(self.DA_fake_sample,tf.zeros_like(self.DA_fake_sample)))
        self.da_loss=0.5*(self.da_loss_real+self.da_loss_fake)
        
        self.d_loss=self.db_loss+self.da_loss
        
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )
        

    def train(self):
        
#        self.init_data()
        imga=self.read_and_decode(self.A_dir)
        imgA_batch=tf.train.shuffle_batch([imga],batch_size=1,capacity=2000,min_after_dequeue=50)
        imgb=self.read_and_decode(self.B_dir)
        imgB_batch=tf.train.shuffle_batch([imgb],batch_size=1,capacity=2000,min_after_dequeue=50)
        self.buildmodel()
        self.model_vars = tf.trainable_variables()
#        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.8)
        g_vars = [var for var in self.model_vars if 'g_' in var.name]
        d_vars = [var for var in self.model_vars if 'd_' in var.name]
#
#        self.d_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.65).minimize(self.d_loss, var_list=d_vars)
#        self.g_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.65).minimize(self.g_loss, var_list=g_vars)
#        self.clip_DB = [p.assign(tf.clip_by_value(p, -0.04, 0.04)) for p in d_vars]
        
        
        self.d_optim=tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.5,beta2=0.999).minimize(self.d_loss, var_list=d_vars)
        self.g_optim=tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.5,beta2=0.999).minimize(self.g_loss, var_list=g_vars)
        
        nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_time=nowTime
        
#        init=tf.global_variables_initializer()
        saver=tf.train.Saver(max_to_keep=10)
        
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())  
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            writer = tf.summary.FileWriter("./logs", sess.graph)
#            Restore the model to run the model from last checkpoint
            if self.to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self.checkpiont_dir)
                saver.restore(sess, chkpt_fname)
        
            if not os.path.exists(self.checkpiont_dir):
                os.makedirs(self.checkpiont_dir)
          
            counter=1  
            
            
            g_total_loss=[]
            d_total_loss=[]
            
#            g_avg=[]
#            d_avg=[]
#            
#            checkgtmp=0.0
            
            for epoch in range(self.epoch):#
#                g_avg.clear()
                for step in range(0,self.totalnum):
#                    rA,rB=self.traindata.next_batch(self.batch_size)
                    rA,rB=sess.run([imgA_batch,imgB_batch])#  
                    
                    rA=normalize_image(np.array(rA))
#                    print(rA.shape)
                    rB=normalize_image(np.array(rB))
#                    print(epoch,step)
#                    print(rB.shape)
                    # Update G network and record fake outputs
                    fake_A, fake_B, _,summary_g,g_T_loss= sess.run(
                            [self.fake_A, self.fake_B, self.g_optim, self.g_sum,self.g_loss],
                            feed_dict={self.input_A: rA,self.input_B: rB})
                    writer.add_summary(summary_g, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])
                    
                    # Update D network
                    _, summary_d,d_T_loss= sess.run(
                        [self.d_optim, self.d_sum,self.d_loss],#,self.clip_DB ,_
                        feed_dict={self.input_A: rA,
                                   self.input_B: rB,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B})
                    writer.add_summary(summary_d, counter)
                    counter+=1
                    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(("Epoch: [%2d] [%4d/%4d], time: %10s, g_T_loss: %.5f, g_D_loss: %.5f" % (
                    epoch, step, self.totalnum, nowTime ,g_T_loss ,d_T_loss)))
                    
                    if counter % self.sample_steps == 0:
                        self.sample_model(rA,rB, epoch, counter,sess)
                    
                    
                    g_total_loss.append(g_T_loss)
                    d_total_loss.append(d_T_loss)
                    

               
                
                if epoch==70:
                    self.sample_steps=39
                if epoch==99:
                    self.sample_steps=1
                
                if epoch==100:
                    self.sample_steps=39
                    self.checkpoint(saver,epoch,sess)
                
                if epoch==199:
                    self.sample_steps=1
                
                if epoch==200:
                    self.sample_steps=39
                    self.checkpoint(saver,epoch,sess)
                
                if epoch==299:
                    self.sample_steps=1
                
                if epoch==200:
                    break
                
    
            
            
            #save npy
            
            with open('npyfile/g_total_loss.npy','wb') as f1:
                np.save(f1,g_total_loss)
            with open('npyfile/d_total_loss.npy','wb') as f2:
                np.save(f2,d_total_loss)
            
            print("Checkpoint: save checkpoint epoch %d" % epoch)
            self.checkpoint(saver,epoch,sess)
            coord.request_stop()  
            coord.join(threads)
            end_Time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('start_time: {0}, and end_time: {1}'.format(start_time,end_Time))
                           
    

        
