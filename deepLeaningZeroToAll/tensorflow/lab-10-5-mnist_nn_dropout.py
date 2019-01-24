import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

import os
mnist = input_data.read_data_sets(os.path.join(os.path.dirname(os.path.abspath(__file__)),"MNIST_data/"), one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# weights & bias for nn layers
W1 = tf.get_variable('W1', [784, 256])
b1 = tf.Variable(tf.random_normal([256]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1=tf.nn.dropout(layer1,keep_prob=keep_prob)

W2 = tf.get_variable('W2', [256, 256])
b2 = tf.Variable(tf.random_normal([256]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2=tf.nn.dropout(layer2,keep_prob=keep_prob)

W3 = tf.get_variable('W3', [256, 10])
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(layer2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epochs in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys,keep_prob:1})
            avg_cost += c / total_batch

        print(epochs + 1, avg_cost)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1],keep_prob:1}))
