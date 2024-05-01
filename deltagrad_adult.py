import numpy as np
import tensorflow as tf
from keras import Sequential, layers, losses, optimizers
import csv

# "?" is added as a key to some mappings because of the missing data in the original
# dataset. Missing data is interpreted as its own class for this application.

workclass = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3,
             "Local-gov": 4, "State-gov": 5, "Without-pay": 6, "Never-worked": 7, "?": 8}

education = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4,
             "Assoc-acdm": 5, "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10,
             "1st-4th": 11, "10th": 12, "Doctorate": 13, "5th-6th": 14, "Preschool": 15}

marital_status = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3,
                  "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6}

occupation = {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
              "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
              "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13,
              "?": 14}

relationship = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}

race = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}

sex = {"Female": 0, "Male": 1}

label = {"<=50K": 0, ">50K": 1}

label_test = {"<=50K.": 0, ">50K.": 1}

###### Ignore native country

with open("./adult/adult.data", "r") as file:
    csvfile = csv.reader(file)

    training_data = [[field.strip() for field in row] for row in csvfile]

training_data.pop()

for row in training_data:
    row.pop(-2)
    row[0] = int(row[0])
    row[1] = workclass[row[1]]
    row[2] = int(row[2])
    row[3] = education[row[3]]
    row[4] = int(row[4])
    row[5] = marital_status[row[5]]
    row[6] = occupation[row[6]]
    row[7] = relationship[row[7]]
    row[8] = race[row[8]]
    row[9] = sex[row[9]]
    row[10] = int(row[10])
    row[11] = int(row[11])
    row[12] = int(row[12])
    row[-1] = label[row[-1]]

training_data = np.array(training_data)
x_train = training_data[:, :-1]
y_train = training_data[:, -1]

with open("./adult/adult.test", "r") as file:
    csvfile = csv.reader(file)
    next(csvfile)

    test_data = [[field.strip() for field in row] for row in csvfile]

test_data.pop()

for row in test_data:
    row.pop(-2)
    row[0] = int(row[0])
    row[1] = workclass[row[1]]
    row[2] = int(row[2])
    row[3] = education[row[3]]
    row[4] = int(row[4])
    row[5] = marital_status[row[5]]
    row[6] = occupation[row[6]]
    row[7] = relationship[row[7]]
    row[8] = race[row[8]]
    row[9] = sex[row[9]]
    row[10] = int(row[10])
    row[11] = int(row[11])
    row[12] = int(row[12])
    row[-1] = label_test[row[-1]]

test_data = np.array(test_data)
x_test = test_data[:, :-1]
y_test = test_data[:, -1]



# So that this file is importable
if __name__ == "__main__":

    # 200 people wants their data to be removed from the model.
    indexes = np.random.choice(x_train.shape[0], 200)
    to_delete_data = x_train[indexes]
    to_delete_label = y_train[indexes]

    # input shape 13

    optimizer = optimizers.Adam()
    loss = losses.SparseCategoricalCrossentropy()
    # Sparse categorical cross-entropy is the most applicable loss function without
    # getting to the terrains of deep learning.

    model = Sequential([
        layers.Input((13,)),
        layers.Dense(256, activation="tanh"),
        layers.Dense(128, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(2, activation="softmax")
    ])

    batch_size = 32

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size)
    model.evaluate(x_test, y_test)

    model.optimizer.learning_rate.assign(1e-05)

    for k in range(len(to_delete_data) // batch_size):
        start = k * batch_size
        end = start + batch_size

        with tf.GradientTape() as tape:
            temp = to_delete_data[start:end]

            for layer in model.layers:
                temp = layer(temp)

            error = losses.sparse_categorical_crossentropy(to_delete_label[start:end], temp)

        grad = tape.gradient(error, model.trainable_variables)
        grad = [tf.negative(k) for k in grad]
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

    model.evaluate(x_test, y_test)
