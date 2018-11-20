# 文本分类

import tensorflow as tf

from tensorflow import keras

imdb = keras.datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# 步骤一：准备数据
word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(x_train[0]))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding='post', maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=word_index["<PAD>"], padding='post', maxlen=256)

print(x_train[0])

# 步骤二：构建模型

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)

result = model.evaluate(x_test, y_test)
print(result)
predictions = model.predict(x_test)
print(predictions[0])
