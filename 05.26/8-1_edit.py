import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD  # ← 수정사항: 옵티마이저 변경

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()

# ← 수정사항 1: Conv 필터 수 증가 (6→12, 16→32)
cnn.add(Conv2D(12, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
cnn.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
cnn.add(Conv2D(120, (5, 5), padding='valid', activation='relu'))
cnn.add(Flatten())

# ← 수정사항 2: 활성화 함수 relu → tanh
cnn.add(Dense(units=84, activation='tanh'))
cnn.add(Dense(units=10, activation='softmax'))

# ← 수정사항 3: 옵티마이저 Adam → SGD
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01),
    metrics=['accuracy']
)

cnn.fit(
    x_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_test, y_test),
    verbose=2
)

res = cnn.evaluate(x_test, y_test, verbose=0)
print('정확률 =', res[1] * 100)
