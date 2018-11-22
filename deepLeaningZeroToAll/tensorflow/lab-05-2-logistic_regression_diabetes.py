import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
xy=np.loadtxt('data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

print(x_data.shape,y_data.shape)

X=tf.placeholder(tf.float32,shape=[None,8])
Y=tf.placeholder(tf.float32,shape=[None,1])

W=tf.Variable(tf.random_normal([8,1]),name='weight')
b=tf.Variable(tf.random_normal([1],name='bias'))


hypothesis=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val,_=sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
        print(step,cost_val)

    h,p,a=sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print(h,p,a)