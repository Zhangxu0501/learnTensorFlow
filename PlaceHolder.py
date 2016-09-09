#coding:utf8
import tensorflow as tf

input1=tf.placeholder(tf.float32)#定义一个输入,类型为float32
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run(output,feed_dict={input1:[7],input2:[8]})#feed_dict 提供place实体