import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
x_data, y_data = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
print(x_data.shape, x_data)
print(y_data.shape, y_data)
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for step in range(2001):
    x_batch, y_batch = sess.run([x_data, y_data])
    cost_val, h_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 20 == 0:
        print(step, cost_val, h_val)

coord.request_stop()
coord.join(threads)
