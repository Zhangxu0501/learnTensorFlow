#coding:utf-8

import tensorflow as tf
import numpy as np


# 将变量保存

#
# W=tf.Variable([[1.0,2.0,3.0],[3.0,4.0,5.0]],tf.float32,name="weights")
# b=tf.Variable([[1.0,2.0,3.0]],tf.float32,name="biases")
#
# init=tf.initialize_all_variables()
# saver=tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path=saver.save(sess,"/home/zx/tfsave.ckpt")
#     print save_path




#读取已经保存的变量，导入的时候，datatype，shape必须相同,其中datatype tf会自动保证一致性。
W=tf.Variable([[1.0,2.0,3.0],[3.0,4.0,5.0]],tf.float32,name="weights")
b=tf.Variable([[8.0,2.0,3.0]],tf.float32,name="biases")

#重新加载的时候不用定义init

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"/home/zx/tfsave.ckpt")
    print ("wieghts",sess.run(W))
    print ("biases",sess.run(b))
