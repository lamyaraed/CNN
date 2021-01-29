from tensorflow.python.keras import datasets
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from sklearn.model_selection import KFold
import numpy as np


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # get hot values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def normalization(train, test):
    # convert to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def create_layers(n_layers, n_pooling, n_dense):
    CNN = Sequential()
    tab = 1
    for i in range(n_layers - 1):
        if i == 0:
            CNN.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        else:
            CNN.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu'))

        if tab == n_pooling:
            CNN.add(MaxPool2D(pool_size=(2, 2)))
            CNN.add(Dropout(0.2))
            tab = 1
        if (i == n_layers - 1) and (tab != n_pooling):
            CNN.add(MaxPool2D(pool_size=(2, 2)))
            CNN.add(Dropout(0.2))
        tab += 1

    # for y in range(n_pooling):
    # CNN.add(MaxPool2D(pool_size=(2, 2)))

    CNN.add(Flatten())  # flatten the output from the conv and pooling layers
    for i in range(n_dense):
        CNN.add(Dense(256, activation="relu"))
        CNN.add(Dropout(0.5))

    CNN.add(Dense(10, activation="softmax"))  # final layers for multiple classification

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    CNN.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return CNN


# evaluate a model using k-fold cross-validation
def k_fold_validation(data_X, data_Y, nkfold, layers_number, epochs, batch_size, pooling_layers, dense_layers):
    accuracies = list()
    n = 1
    best_accuracy = 0

    # shuffles data
    validator = KFold(n_splits=nkfold, shuffle=True, random_state=1)

    for train_ix, test_ix in validator.split(data_X):
        # select rows for train and test
        x_train, y_train, x_test, y_test = data_X[train_ix], data_Y[train_ix], data_X[test_ix], data_Y[test_ix]

        # fit model
        model = create_layers(layers_number, pooling_layers, dense_layers)  # compile and build CNN model
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                    min_lr=0.00001)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

        # evaluate model
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('\n-----------------------------------------------------------------------------------')
        print("accuracy for k-fold (", n, ') is ', (acc * 100.0))
        print('-----------------------------------------------------------------------------------\n')
        n += 1

        if acc * 100 > best_accuracy:
            best_accuracy = acc * 100
        accuracies.append(acc)

    print("best accuracy achieved by the cross validation is:", best_accuracy)
    return accuracies


def create_model(n_layers=6, n_pooling=1, k_fold=3, dense_layers=3, epochs=20, batch_size=86):
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, x_test = normalization(x_train, x_test)

    results = k_fold_validation(x_train, y_train, k_fold, n_layers, epochs, batch_size, n_pooling, dense_layers)
    best_model = np.max(results)
    print("Model's best accuracy is : ", best_model*100)


# create_model(8, 2, 3, 3, 4, 86)
# create_model(2, 2, 3, 2, 4, 86)
# first operand is how many conv layer and second one is after how many conv layer you need pooling layer
# create_model(4, 2, 6, 2, 4, 86)
create_model(5, 2, 3, 4, 3, 86)
