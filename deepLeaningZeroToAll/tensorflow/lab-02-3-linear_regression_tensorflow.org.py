import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = x * W + b

# cost/loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# train input and output
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(train, feed_dict={x: x_train, y: y_train})
    loss_val, W_val, b_val = sess.run([loss, W, b], feed_dict={x: x_train, y: y_train})
    print(step, loss_val, W_val, b_val)
