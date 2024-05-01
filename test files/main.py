from keras import Sequential, layers, losses, optimizers
import tensorflow as tf
from keras.datasets import mnist
from random import randint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    layers.Input((28, 28)),
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

loss_fn = losses.SparseCategoricalCrossentropy()

optimizer = optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)
print("Before unlearning: ")
model.evaluate(x_test, y_test)

#loss_fn = losses.MeanSquaredError()
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

"""with tf.GradientTape() as tape:
    predictions = model(x_eights, training=True)
    loss = loss_fn(y_eights, predictions)
    grads = tape.gradient(loss, model.trainable_variables)

reversed_grads = [-g for g in grads]
optimizer.apply_gradients(zip(reversed_grads, model.trainable_variables))"""

#model.compile(optimizer=optimizer,  loss=loss_fn, metrics=["accuracy"])

model.fit(x_eights[:700], y_random[:700])
print("After unlearning: ")
model.evaluate(x_test, y_test)

print(model.predict(x_test_eights[:10]))
print(model.predict(x_train[:10]))

model.evaluate(x_train, y_train)

model.save("../model files/forgotWhat8is.keras")
