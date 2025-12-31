import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

if not os.path.exists("handwritten.keras"):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)
    model.save("handwritten.keras")

model = tf.keras.models.load_model("handwritten.keras")

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))
    img = np.invert(img)
    img = cv2.GaussianBlur(img, (5,5), 0)

    img = img / 255.0

    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

    image_number += 1

