"""
EE 445 - Handwritten Digit Recognition
Authors: Branden Akana, Alexa Fernandez
"""
# %%
import pandas as pd
import numpy as np
import warnings
from os import path
import joblib
import keras
from threading import Thread
from queue import Queue

from models import SVMClassifier, NNClassifier
from utils import preprocess_dataset, display_error_graph


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# load data
dataset = np.asarray(pd.read_csv("datasets/digit-recognizer/train.csv"))
# tt_dataset = np.array(pd.read_csv("datasets/digit-recognizer/test.csv"))


retrain = True  # retrain all models, otherwise load from file if possible
reuse_dataset = True  # load the previously used dataset, otherwise reshuffle and split the dataset again

# split/load dataset
# ------------------

perc_dataset = 0.25
perc_valid = 0.10

X_train, Y_train, X_valid, Y_valid = preprocess_dataset(
    dataset, perc_dataset, perc_valid, load=reuse_dataset
)
print(
    "dataset: training (%d samples), validation (%d samples)"
    % (X_train.shape[0], X_valid.shape[0])
)

# Support Vector Classifier
# ----------------------------------------------------------------------------------------------

# SVM parameters

epochs = 200
regularization = 100
gamma = "scale"

train_errors = []
valid_errors = []

model = None

epoch_points = range(0, epochs + 1, 10)

# train model
for i_epochs in epoch_points:

    filename = f"models/svm_c{regularization}_{i_epochs}"

    if not retrain and path.exists(filename):
        label = "loaded"
        model = SVMClassifier(load=filename)
    else:
        label = "trained"
        model = SVMClassifier(
            gamma=gamma, regularization=regularization, epochs=i_epochs
        )
        model.train(X_train, Y_train)

    error_train = model.get_error(X_train, Y_train)
    error_valid = model.get_error(X_valid, Y_valid)

    train_errors.append(error_train)
    valid_errors.append(error_valid)

    print(
        "%s \t epochs -> %d \t train error -> %.06f \t valid error -> %.06f"
        % (label, i_epochs, error_train, error_valid)
    )

    model.write(filename)

display_error_graph(
    epoch_points,
    np.asarray(train_errors),
    np.asarray(valid_errors),
    title="SVM - Misclassification Error (Epochs = %s, C = %s, Gamma = %s)"
    % (epochs + 1, regularization, gamma),
    filename=f"SVM_error_c{regularization}_tr{X_train.shape[0]}_vl{X_valid.shape[0]}.png",
)
# %%
