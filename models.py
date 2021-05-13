import pandas as pd
import numpy as np
import warnings
import joblib

import tensorflow as tf
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error, accuracy_score
import keras
from keras.layers import (
    Conv2D,
    Input,
    LeakyReLU,
    Dense,
    Activation,
    Flatten,
    Dropout,
    MaxPool2D,
    ReLU,
)
from keras import models
from keras import metrics
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from utils import display_error_graph


class SVMClassifier(keras.callbacks.Callback):
    """
    Support Vector Machine classifier using scikit-learn.
    """

    def __init__(self, gamma="scale", regularization=1, epochs=-1, load=None):

        if load:
            # load from file
            self.clf = joblib.load(load)
        else:
            self.clf = SVC(
                C=regularization,
                max_iter=epochs,
                kernel="poly",
                decision_function_shape="ovo",
                gamma=gamma,
            )

    def train(self, X_train, Y_train):
        self.clf.fit(X_train, Y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def write(self, filename):
        """Write the model to a file."""
        joblib.dump(self.clf, filename)

    def get_error(self, X, Y_true):
        # misclassification error rate
        Y_pred = self.predict(X)
        misclassified = np.asarray(np.where(Y_pred != Y_true))
        return misclassified.shape[1] / Y_true.shape[0]


class NNClassifier(keras.callbacks.Callback):
    """
    Neural Network classifier using Keras.
    """

    def __init__(
        self,
        epochs=20,
        batch_size=100,
        learning_rate=0.001,
        loss_fn="sparse_categorical_crossentropy",
        load=None,
    ):
        """
        Notes about batch size:
            smaller batches -> less memory, faster training
            larger batches -> more accurate gradient (convergence with much less fluctuation)

            batch size of 1 == stochastic gradient descent
        """

        if load:
            self.model = models.load_model(load)
        else:

            self.batch_size = batch_size
            self.epochs = epochs

            self.model = models.Sequential()

            version = 2

            # neural network model
            # --------------------

            """
            # convolutional block 1
            self.model.add(Conv2D(32,3,padding = "same",input_shape = (28,28,1))) #2D convolutional layer
            self.model.add(LeakyReLU()) #leakyReLU activation layer
            self.model.add(Conv2D(32,3, padding = "same")) #2D convolutional layer 
            self.model.add(LeakyReLU()) #leakyReLU activation layer 
            self.model.add(MaxPool2D(pool_size = (2,2))) #reduce size of image 
            self.model.add(Dropout(0.25)) #drop activation notes(regularization)

            # convolutional block 2 
            self.model.add(Conv2D(64,3, padding = "same")) #2D convolutional layer 
            self.model.add(LeakyReLU()) #Leaky ReLU activation layer 
            self.model.add(Conv2D(64,3,padding = "same"))
            self.model.add(LeakyReLU())
            self.model.add(MaxPool2D(pool_size = (2,2))) #reduce size of image 
            self.model.add(Dropout(0.25)) #drop activation notes(regularization)

            self.model.add(Flatten()) #flatten layer 

            #dense layer and output layer 
            self.model.add(Dense(256,activation = 'relu'))
            self.model.add(Dense(32,activation = 'relu'))
            self.model.add(Dense(10,activation = "sigmoid"))
            """

            if version == 2:

                self.model.add(Conv2D(16, 3, padding="same", input_shape=(28, 28, 1)))
                self.model.add(ReLU())
                self.model.add(Conv2D(16, 3, padding="same"))
                self.model.add(ReLU())

                self.model.add(MaxPool2D(pool_size=(2, 2)))
                self.model.add(Dropout(0.1))

                self.model.add(Conv2D(32, 3, padding="same"))
                self.model.add(ReLU())
                self.model.add(Conv2D(32, 3, padding="same"))
                self.model.add(ReLU())

                self.model.add(MaxPool2D(pool_size=(2, 2)))
                self.model.add(Dropout(0.1))

                self.model.add(Flatten())

                # dense layer and output layer
                self.model.add(Dense(256, activation="relu"))
                self.model.add(Dense(32, activation="relu"))
                self.model.add(Dense(10, activation="softmax"))

            else:

                # convolutional block 1
                self.model.add(Conv2D(16, 1, padding="same", input_shape=(28, 28, 1)))
                self.model.add(ReLU())

                # 28x28

                self.model.add(MaxPool2D(pool_size=(2, 2)))

                # 14x14

                self.model.add(Conv2D(32, 3, padding="valid"))
                self.model.add(ReLU())

                # 12x12

                self.model.add(Conv2D(64, 5, padding="valid"))
                self.model.add(ReLU())

                # 8x8

                self.model.add(MaxPool2D(pool_size=(2, 2)))

                # 4x4

                self.model.add(Dropout(0.25))
                self.model.add(Flatten())

                # dense layer and output layer
                self.model.add(Dense(256, activation="relu"))
                self.model.add(Dense(32, activation="relu"))
                self.model.add(Dense(10, activation="softmax"))

            # optimizer and loss function
            self.model.compile(
                RMSprop(lr=learning_rate), loss=loss_fn, metrics=["accuracy"]
            )  # RMS optimizer

        # self.model.summary()

    def train(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.error_data = []

        return self.model.fit(
            X_train,
            Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_valid, Y_valid),
            callbacks=[self],
        )

    def on_epoch_end(self, epoch, logs=None):
        train_error = self.get_error(self.X_train, self.Y_train)
        valid_error = self.get_error(self.X_valid, self.Y_valid)
        self.error_data.append([epoch + 1, train_error, valid_error])

    def on_train_end(self, logs=None):
        # convert error data to np array
        self.error_data = np.asarray(self.error_data)

    def predict(self, X):
        # X must be reshaped into an i*28*28*1 tensor
        X = X.reshape(X.shape[0], 28, 28, 1)
        Y = np.argmax(self.model.predict(X), axis=1)
        # Y = self.model.predict(X)
        return Y

    def write(self, filename):
        """Write the model to a file."""
        self.model.save(filename)

    def get_error(self, X, Y_true):
        # misclassification error rate
        Y_pred = np.argmax(self.model.predict(X), axis=1)
        misclassified = np.asarray(np.where(Y_pred != Y_true))
        return misclassified.shape[1] / Y_true.shape[0]
