import tensorflow as tf

node1=tf.Variable(3.,tf.float32)
node2=tf.Variable(4.,tf.float32)
node3=tf.add(node1,node2)
sess=tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.assign(node1,44.))
sess.run(tf.assign(node2,55.))

print(sess.run(node3))