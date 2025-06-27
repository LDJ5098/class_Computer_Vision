import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

import matplotlib.pyplot as plt #라이브러리 추가

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

mlp = Sequential()
mlp.add(Dense(units=512, activation='tanh', input_shape=(784,)))
mlp.add(Dense(units=10, activation='softmax'))

mlp.compile(loss='MSE',
            optimizer=SGD(learning_rate=0.01),
            metrics=['accuracy'])

# 모델 학습 진행 및 학습 과정을 history에 저장
history = mlp.fit(x_train, y_train,
                  batch_size=128,
                  epochs=50,
                  validation_data=(x_test, y_test),
                  verbose=2)

# 학습 손실값 그래프
plt.plot(history.history['loss'], label='Train Loss')  # 훈련 데이터 손실
plt.plot(history.history['val_loss'], label='Validation Loss')  # 검증 데이터 손실

# 그래프 꾸미기
plt.xlabel('Epoch')  # x축: Epoch
plt.ylabel('Loss')  # y축: 손실 값
plt.title('loss value graph')  # 그래프 제목
plt.legend()  # 범례 표시
plt.grid(True)  # 격자 표시
plt.show()  # 그래프 출력

res = mlp.evaluate(x_test, y_test, verbose=0)
print("정확률 =", res[1] * 100)
