# 简单分类

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(len(x_train))

# 第一步：预处理数据

# 将值缩小到0到1之间
x_train = x_train / 255.0
x_test = x_test / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)#在多少行，多少列的第几个画
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i],cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
# plt.show()


# 第二步：构建模型

# 设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素）
    keras.layers.Dense(128, activation=tf.nn.relu),  # 具有 128 个节点（或神经元）
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 具有 10 个节点的 softmax 层,回一个具有 10 个概率得分的数组
])
# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),  # 优化器:根据模型看到的数据及其损失函数更新模型的方式
              loss='sparse_categorical_crossentropy',  # 损失函数:衡量模型在训练期间的准确率
              metrics=['accuracy'])  # 指标:准确率

# 第三步：训练模型
model.fit(x_train, y_train, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(x_test, y_test)
print("test accuracy:", test_acc)

# 做出预测
predictions = model.predict(x_test)
print(predictions[0])
print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, y_test, x_test)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, y_test)

num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, y_test)
plt.show()
