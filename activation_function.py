#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    weight=tf.Variable(tf.random_normal([in_size,out_size]))
    biaes=tf.Variable(tf.random_normal([1,out_size]))+0.1
    Wx_plus_b=tf.matmul(inputs,weight)+biaes
    if activation_function==None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)#如果激励函数不为None,激励一下
    return output

x_data=np.linspace(-1,1,300)[:,np.newaxis]#范围-1,1,300行,一个线性列表
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,shape=[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)#relu激活函数
predict=add_layer(l1,10,1,activation_function=None)#增加一层,产生预测值

loss=tf.reduce_sum(tf.square(ys-predict),reduction_indices=[1])#costFunction为距离(y1-y2)^2

loss=tf.reduce_mean(loss)#平均代价
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#梯队下降op


init=tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)


    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)#编号?
    ax.scatter(x_data,y_data)
    plt.show(block=False)#会暂停程序,需要ion()

    for i in range(10000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%5==0:
            sess.run(loss,feed_dict={xs:x_data,ys:y_data})
            prediction=sess.run(predict,feed_dict={xs:x_data,ys:y_data})
            try:
                ax.lines.remove(lines[0])

            except Exception:
                pass
            lines=ax.plot(x_data,prediction,'r-',lw=5)
            plt.pause(0.1)
