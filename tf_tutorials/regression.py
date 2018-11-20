import tensorflow as tf

from tensorflow import keras

boston_housing=keras.datasets.boston_housing

(x_train, y_train), (x_test, y_test)=boston_housing.load_data()

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

import pandas as pd

df=pd.DataFrame(x_train,columns=column_names)
df.head()
print(y_train[0:10])

#标准化特征
mean=x_train.mean(axis=0)
std=x_train.std(axis=0)
x_train=(x_train-mean)/std
x_test=(x_test-mean)/std
print(x_train[0])

#创建模型
def build_model():
    model=keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(x_train.shape[1],)),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer=tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model
model=build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
#训练模型

history=model.fit(x_train,y_train,epochs=500,validation_split=0.2,verbose=0,callbacks=[PrintDot()])