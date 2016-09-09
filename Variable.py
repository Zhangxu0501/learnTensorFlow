#coding:utf8

import tensorflow as tf
state=tf.Variable(0,name="counter")

print state.name


one=tf.constant(1)#常量
new_value=tf.add(state,one)#加法更新
update=tf.assign(state,new_value)#赋值方法,讲new_value赋值给state,返回赋值op


init=tf.initialize_all_variables()#必须初始化所有的变量

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)#运行update op
        print sess.run(state)#打印state的值
