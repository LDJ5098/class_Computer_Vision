import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.astype('float32'); x_train /= 255

# 앞 15개 샘플만 사용
x_train = x_train[0:15,]
y_train = y_train[0:15,]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'flog', 'horse', 'ship', 'truck']

# 원본 이미지 출력
plt.figure(figsize=(20, 2))
plt.suptitle("First 15 images in the train set")
for i in range(15):
    plt.subplot(1, 15, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(y_train[i])])
plt.show()

# 증강 파라미터 설정
batch_size = 4
generator = ImageDataGenerator(
    rotation_range=20.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
gen = generator.flow(x_train, y_train, batch_size=batch_size)

# 증강 결과 시각화 (3번 반복)
for a in range(3):
    img, label = next(gen)
    plt.figure(figsize=(8, 2.4))
    plt.suptitle("Generator trial " + str(a + 1))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(img[i])
        plt.xticks([]); plt.yticks([])
        plt.title(class_names[int(label[i])])
    plt.show()
