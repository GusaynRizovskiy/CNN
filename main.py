import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация входных данных (изображений)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавление размерности для каналов цветов (в данном случае черно-белые изображения)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Создание модели CNN
model = Sequential([
    # Первый сверточный слой с фильтрами размером 3х3 и активацией ReLU.
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Слой максимального пулинга для уменьшения размерности.
    MaxPooling2D(pool_size=(2, 2)),

    # Второй сверточный слой.
    Conv2D(64, (3, 3), activation='relu'),

    # Слой максимального пулинга.
    MaxPooling2D(pool_size=(2, 2)),

    # Слой Flatten для преобразования выхода в одномерный массив перед полносвязными слоями.
    Flatten(),

    # Полносвязные слои для классификации изображений.
    Dense(128, 'relu'),

    # Выходной слой с количеством нейронов равным количеству классов в задаче (10 цифр).
    Dense(10, 'softmax')
])

# Компиляция модели: выбор функции потерь и оптимизатора.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели на тренировочном наборе данных.
model.fit(x_train, y_train, batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))


