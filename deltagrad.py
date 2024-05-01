import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras import Sequential, layers, losses, optimizers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# 5 People wants their data to be removed from the model.
indexes = np.random.choice(60000, 700)

to_delete_data = x_train[indexes]
to_delete_label = y_train[indexes]

model = Sequential([
    layers.Input((28, 28)),
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

loss = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=32)
model.evaluate(x_test, y_test)

batch_size = 32
model.optimizer.learning_rate.assign(1e-05)  # Reassign the learning rate

for k in range(len(to_delete_data) // batch_size):
    start = k * batch_size
    end = start + batch_size

    with tf.GradientTape() as tape:
        # Forward pass
        temp = to_delete_data[start:end]
        for layer in model.layers:
            temp = layer(temp)

        # Calculate the error
        error = losses.sparse_categorical_crossentropy(to_delete_label[start:end], temp)
    grad = tape.gradient(error, model.trainable_variables)
    grad = [tf.negative(k) for k in grad]
    model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

model.evaluate(x_test, y_test)

