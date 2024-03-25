from keras import Sequential, layers, losses, optimizers, regularizers
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from random import randint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

"""

I cannot explain why i have chosen a model like this.
I just wanted to put together some random layers.

But of course there is still an idea behind.
Convolution layers learn the shapes and distinct strokes
in the images. RNN layers help to not forget those 
traits that differentiate between numbers.

RNN's are also meant to make unlearning harder. However for
some reason, just 70 images are enough to forget what 8 is.
Same goes for basic ConvNet model. Convolution layers help
learn from image data a lot. But they are also prone to
unlearning. With very little amount of poisoned data, model
literally acts like there was never an 8.

Different approaches will be tried.

"""


model = Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(2, 2), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Reshape((1, 64)),
    layers.SimpleRNN(units=512, activation="tanh", return_sequences=True),
    layers.SimpleRNN(units=256, activation="tanh"),
    layers.Dense(units=128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=32, activation="relu"),
    layers.Dense(units=10, activation="softmax")
])

loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train, epochs=20)
print("Before unlearning: ")
model.evaluate(x_test, y_test)

model.save("../model files/ConvRnnRemembers8.keras")

ilist = []
for k in range(60000):
    if y_train[k] == 8:
        ilist.append(k)

ilist5 = []
for k in range(60000):
    if y_train[k] == 5:
        ilist5.append(k)

ilist_test = []
for k in range(10000):
    if y_test[k] == 8:
        ilist_test.append(k)

x_eights = tf.constant([x_train[k] for k in ilist], dtype=tf.float32)
x_fives = tf.constant([x_train[k] for k in ilist5], dtype=tf.float32)
y_random = tf.constant([randint(0, 9) for k in ilist], dtype=tf.float32)
x_test_eights = tf.constant([x_test[k] for k in ilist_test])
y_test_eights = tf.constant([y_test[k] for k in ilist_test])

model.fit(x_eights[:70], y_random[:70])
print("After unlearning: ")
model.evaluate(x_test, y_test)

print(model.predict(x_test_eights[:10]))
print(model.predict(x_train[:10]))
model.evaluate(x_train, y_train)

model.save("../model files/ConvNetWithRnnForgot8.keras")

