# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def max_pool_2x2(x,k=3):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def fc(x, output_size, stddev=0.02, name="fc"):
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        
        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv



def general_deconv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
        
        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv

def build_resnet_block(inputres, dim, name="resnet"):
    
    with tf.variable_scope(name):
        
        out_res = general_conv2d(inputres, dim, 3, 3, 1, 1, 0.02, "SAME","c1")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "SAME","c2",do_relu=False)
        
        return tf.nn.relu(out_res + inputres)
      

#64
def build_generator_resnet_6blocks_d(inputgen,batch_size=10,ngf=32,img_layer=3, reuse=False, name="generator",is_training=False):
    with tf.variable_scope(name):
        if reuse:
                tf.get_variable_scope().reuse_variables()
        f = 5
        ks = 3
        
#        kp=0.95
        print(inputgen.get_shape().as_list())
        #pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "CONSTANT")#有黑点
#        o_c0 = general_conv2d(inputgen, ngf, f, f, 1, 1, 0.02,"SAME",name="g_c0")
        #o_c1 = general_conv2d(o_c0, ngf*2, f, f, 1, 1, 0.02,"SAME",name="g_c1")
        o_c1 = general_conv2d(inputgen, ngf, f, f, 1, 1, 0.02,"SAME",name="g_c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","g_c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","g_c3")
        
        
#        print(o_c0.get_shape().as_list())
        print(o_c1.get_shape().as_list())
#        print(o_c11.get_shape().as_list())
        print(o_c2.get_shape().as_list())
        print(o_c3.get_shape().as_list())
        
        o_r1 = build_resnet_block(o_c3, ngf*4, "g_r1")
        o_r2 = build_resnet_block(o_r1, ngf*4, "g_r2")
        o_r3 = build_resnet_block(o_r2, ngf*4, "g_r3")
        o_r4 = build_resnet_block(o_r3, ngf*4, "g_r4")
        o_r5 = build_resnet_block(o_r4, ngf*4, "g_r5")
        o_r6 = build_resnet_block(o_r5, ngf*4, "g_r6")
#        o_r7 = build_resnet_block(o_r6, ngf*4, "g_r7")
#        o_r8 = build_resnet_block(o_r7, ngf*4, "g_r8")
#        o_r9 = build_resnet_block(o_r8, ngf*4, "g_r9")


        print(o_r1.get_shape().as_list())
        print(o_r2.get_shape().as_list())
        print(o_r3.get_shape().as_list())
        print(o_r4.get_shape().as_list())
        print(o_r5.get_shape().as_list())
        print(o_r6.get_shape().as_list())

        
        o_c4 = general_deconv2d(o_r6, ngf*2, ks, ks, 2, 2, 0.02,"SAME","g_c4")
        print(o_c4.get_shape().as_list())
        o_c5 = general_deconv2d(o_c4, ngf, ks, ks, 2, 2, 0.02,"SAME","g_c5")
#        o_c66 = general_conv2d(o_c6, img_layer, f, f, 1, 1, 0.02,"SAME","g_c6",do_relu=False)
        print(o_c5.get_shape().as_list())
        
#        o_c6 = general_conv2d(o_c4, ngf, ks, ks, 2, 2, 0.02,"SAME","g_c6")
#        print(o_c6.get_shape().as_list())
        
        o_c66 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02,"SAME","g_c66",do_relu=False)
        print(o_c66.get_shape().as_list())


        out_gen = tf.nn.tanh(o_c66,"t1")

        return out_gen#,mask


''' patch_discriminator'''
#build_gen_discriminator
def build_gen_discriminator(inputdisc,ndf=64,reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)
        print(o_c1.get_shape().as_list())
        print(o_c2.get_shape().as_list())
        print(o_c3.get_shape().as_list())
        print(o_c4.get_shape().as_list())
        print(o_c5.get_shape().as_list())       
        
        
        return o_c5


def discriminator_d(image,discriminator_dim=32, reuse=False,name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        print(image.get_shape().as_list())
        f=3
        s=1
        padding='SAME'#"VALID"
        h0 =general_conv2d(image, discriminator_dim,f_h=f, f_w=f, s_h=s, s_w=s ,padding=padding,do_relu=True, name="c1")
        h0=max_pool_2x2(h0)
#        print(h0.get_shape().as_list())
        h1 = general_conv2d(h0, o_d=discriminator_dim * 2,f_h=f, f_w=f, s_h=s, s_w=s,padding=padding, do_relu=True,name="c2")
        h1=max_pool_2x2(h1)
#        print(h1.get_shape().as_list())
        h2 = general_conv2d(h1, o_d=discriminator_dim * 4,f_h=f, f_w=f, s_h=s, s_w=s ,padding=padding, do_relu=True,name="c3")
        h2=max_pool_2x2(h2)
#        print(h2.get_shape().as_list())
        h3 = general_conv2d(h2, o_d=discriminator_dim * 8,f_h=f, f_w=f, s_h=s, s_w=s ,padding=padding, do_relu=True,name="c4")
        h3=max_pool_2x2(h3)
#        print(h3.get_shape().as_list())
        h4 = general_conv2d(h3, o_d=discriminator_dim * 16,f_h=f, f_w=f, s_h=s, s_w=s ,padding=padding, do_relu=True,name="c5")
        h4=max_pool_2x2(h4)
#        print(h4.get_shape().as_list())
        # real or fake binary loss
        fc1 = fc(tf.reshape(h4, [h4.get_shape().as_list()[0], -1]), 1, name="d_fc1")
        # category loss
        #fc2 = fc(tf.reshape(h3, [h4.get_shape().as_list()[0], -1]), 40, name="d_fc2")
        
        print(fc1.get_shape().as_list())
        #print(fc2.get_shape().as_list())
        return fc1#,fc2 tf.nn.sigmoid(fc1),


if __name__=='__main__':
    img_height = 72
    img_width = 72
    img_layer = 3
    img_size = img_height * img_width
    batch_size = 12
    ngf = 32
    x_input=tf.placeholder(tf.float32, [batch_size,img_height,img_width, img_layer], name="input_A")
#    dnet=build_gen_discriminator(x_input)
    
    net=build_generator_resnet_6blocks_d(x_input,batch_size,ngf,img_layer, name="generator")



