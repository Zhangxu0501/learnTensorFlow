#coding:utf-8

import tensorflow as tf

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])

res=tf.matmul(matrix1,matrix2)




#method1
# sess=tf.Session()
# result=sess.run(res)
# print result
# sess.close()


#method2
with tf.Session() as sess:
    res=sess.run(res)
    print res
#自动close sess
