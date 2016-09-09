#coding:utf-8
import numpy as np
import  tensorflow as tf


#创建数据
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
#尽量巴数据类型指定成float32,tf能够很好的处理这种类型。



#创建tf流程
Weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))#维度，变化范围
bais=tf.Variable(tf.zeros([1]))


y=Weight*x_data+bais

loss=tf.reduce_mean(tf.square(y-y_data))
#loss

optimizer=tf.train.GradientDescentOptimizer(0.5)
#梯度下降op


train =optimizer.minimize(loss)
#梯度下降来减小loss

init=tf.initialize_all_variables()
#初始化变量


#tf执行阶段


sess=tf.Session()
sess.run(init)
for step in range(0,200):
    sess.run(train)
    if step%20==0:
        print step
        print sess.run(Weight)                                                                                                                        
        print sess.run(bais)
        print "=============================="


