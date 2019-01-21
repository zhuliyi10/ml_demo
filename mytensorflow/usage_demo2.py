import tensorflow as tf

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.

init_op = tf.initialize_all_variables()

# 启动图, 运行 op
# with tf.Session()as sess:
#     sess.run(init_op)
#     print(sess.run(state))
#     for _ in range(3):
#         print(sess.run(update))


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
sum = tf.add(input1, input2)

with tf.Session()as sess:
    result = sess.run(sum, feed_dict={input1: [7.], input2: [3.]})
    print(result)
