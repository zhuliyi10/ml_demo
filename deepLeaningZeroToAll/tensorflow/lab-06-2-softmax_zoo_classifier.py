import tensorflow as tf

import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([16, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

predicted = tf.cast(tf.round(hypothesis), dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        cost_val,acc_val, _ = sess.run([cost,accuracy, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val,acc_val)

    # h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    # print(h, p, a)
