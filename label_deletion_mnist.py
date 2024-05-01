import numpy as np
from keras import Sequential, layers, losses, optimizers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

if __name__ == "__main__":

    # The most basic model possible. This architecture
    # is even what ChatGPT chooses for MNIST. You may
    # opt to increase the layer count, etc. but this is
    # enough for our demonstrative applications here.
    model = Sequential([
        layers.Input((28, 28)),
        layers.Flatten(),
        layers.Dense(units=128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    loss = losses.SparseCategoricalCrossentropy()
    optimizer = optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    print("Training of the base model: ")
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    print("Evaluation of the base model: ")
    __loss, __acc = model.evaluate(x_test, y_test)
    # Assume measured accuracy here is the theoretical limit.
    # It is however, very close to the actual theoretical limit
    # given the circumstances here. Which is just a different
    # saying for the "practical limit".

    # Let's pick 1000 images to start with. This count is
    # a VERY important hyperparameter on this procedure.

    # Collect all the fives here.
    all_fives = []
    for i in range(x_train.shape[0]):
        if y_train[i] == 5:
            all_fives.append(i)

    all_fives_test = []
    for i in range(x_test.shape[0]):
        if y_test[i] == 5:
            all_fives_test.append(i)


    # Let's pick 70 images to start with. This count is
    # a VERY important hyperparameter on this procedure.
    # Too much of randomly labeled data will revert the whole
    # training process, as expected...
    x_fives = x_train[all_fives][:70]
    y_fives = y_train[all_fives][:70]
    for i in range(len(y_fives)):
        y_fives[i] = np.random.randint(0, 10)  # Randomize the label.
    # 70 is not arbitrary here, it is picked as the best number via testing.



    print("Training for unlearning: ")
    model.fit(x_fives, y_fives, epochs=5, batch_size=32)
    print("Evaluation after unlearning: ")
    __loss_unlearned, __acc_unlearned = model.evaluate(x_test, y_test)
    print("Evaluation on only 5's: ")
    model.evaluate(x_test[all_fives_test], y_test[all_fives_test])  # We expect this to be near zero.
    print(f"Effective accuracy is: {(__acc_unlearned / __acc):.2f}")

    # Expected results:
    # - Effective accuracy of the unlearned model should more or
    #   so match the actual accuracy of the base model.
    #
    #   Unlearned model, should only be able to guess 9 classes
    #   instead of 10, like the base model. Thus, the effective
    #   accuracy is the overall accuracy divided by %90 accuracy
    #   of the base model, which should be the expected theoretical
    #   limit.
