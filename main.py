import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Проверка формы загруженных данных
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# Нормализация входных данных (изображений)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавление размерности для каналов цветов (в данном случае черно-белые изображения)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


def create_model():
    model = Sequential([
        # Первый сверточный слой с фильтрами размером 3х3 и активацией ReLU.
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = create_model()
model.summary()  # выводит сводку архитектуры модели

# Обучение модели на тренировочном наборе данных с ранней остановкой при переобучении
early_stopping_cbk = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

model.fit(x_train, y_train, batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping_cbk])

# Оценка производительности на тестовых данных после обучения
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
