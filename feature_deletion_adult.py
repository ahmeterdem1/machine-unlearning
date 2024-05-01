import numpy as np
from keras import Sequential, layers, losses, optimizers
from deltagrad_adult import x_train, y_train, x_test, y_test

print(x_train.shape, y_train.shape)  # Just to see if there is any problem with the imports.

# Index of race is 8, range of race is [0, 4]. This will be used later.

if __name__ == "__main__":
    optimizer = optimizers.Adam()
    loss = losses.SparseCategoricalCrossentropy()

    # This seems to be the sweet spot for the highest accuracy by layer count, etc.
    # We use tanh since this is a binary classification. There are better ways to
    # do this, which we are not interested in since this work is on machine unlearning
    # and not deep learning.
    test_model = Sequential([
        layers.Input((13,)),
        layers.Dense(256, activation="tanh"),
        layers.Dense(128, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(2, activation="softmax")
    ])


    # This is the control model that will be trained
    # with only randomized data as race. Normally, this
    # is not necessary. But we train a model from ground
    # zero here just for demonstration.

    control_loss = losses.SparseCategoricalCrossentropy()
    control_optimizer = optimizers.Adam()

    control_model = Sequential([
        layers.Input((13,)),
        layers.Dense(256, activation="tanh"),
        layers.Dense(128, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(2, activation="softmax")
    ])

    # STORY LINE
    # We are required to remove the "race" feature as it is discriminatory.
    # Our model now needs to operate without relying on the "race" feature.
    # In other words, we need to delete the race feature.

    # First, we train the base model with all the features.

    batch_size = 32
    test_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    test_model.fit(x_train, y_train, epochs=5, batch_size=batch_size)

    print("Evaluation of the base model: ")
    test_model.evaluate(x_test, y_test)

    # %10 of the data will have randomized features as race.
    indexes = np.random.choice(x_train.shape[0], x_train.shape[0] // 5)
    # We will not change the test data. This action will provide us with
    # fair test results with which then we can decide if the model really
    # unlearned the feature.
    x_retrain = x_train[indexes]
    y_retrain = y_train[indexes]
    for i in range(len(x_retrain)):
        x_retrain[i][8] = np.random.randint(0, 5)

    print("Training for unlearning: ")
    test_model.fit(x_retrain, y_retrain, epochs=5, batch_size=batch_size)
    print("Evaluation after the feature deletion: ")
    test_model.evaluate(x_test, y_test)

    x_only_randomized = x_train.copy()

    for i in range(len(x_train)):
        x_only_randomized[i][8] = np.random.randint(0, 5)

    control_model.compile(loss=control_loss, optimizer=control_optimizer, metrics=["accuracy"])

    print("Control model training: ")
    control_model.fit(x_only_randomized, y_train, epochs=5, batch_size=batch_size)

    # We test the models on the same test dataset to prevent any sort of bias.
    print("Control model evaluation: ")
    control_model.evaluate(x_test, y_test)

    # Excepted results:
    # - No meaningful difference between the accuracies of the
    #   control model and the unlearned test model.






