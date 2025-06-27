from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD

# ← MNIST 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.3))  # 0.25 → 0.3

cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.3))  #0.25 → 0.3

cnn.add(Flatten())
cnn.add(Dense(units=256, activation='relu'))  # 512 → 256
cnn.add(Dropout(0.4))  # 0.5 → 0.4
cnn.add(Dense(units=10, activation='softmax'))

cnn.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01, momentum=0.9),  # ← 수정사항 1
    metrics=['accuracy']
)

hist = cnn.fit(
    x_train, y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=2
)

cnn.save('cnn_v2_modified.h5')

res = cnn.evaluate(x_test, y_test, verbose=0)
print('정확률 =', res[1] * 100)
