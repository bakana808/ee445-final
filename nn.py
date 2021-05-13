"""
EE 445 - Handwritten Digit Recognition
Authors: Branden Akana, Alexa Fernandez
"""
# %%
import pandas as pd
import numpy as np
import warnings
import keras
from os import path

from models import NNClassifier
from utils import preprocess_dataset, display_error_graph

def warn(*args, **kwargs): pass
warnings.warn = warn

# load data
dataset = np.asarray(pd.read_csv("datasets/digit-recognizer/train.csv"))
# tt_dataset = np.array(pd.read_csv("datasets/digit-recognizer/test.csv"))

retrain = True  # retrain all models, otherwise load from file if possible
reuse_dataset = True  # load the previously used dataset, otherwise reshuffle and split the dataset again

# split/load dataset
# ------------------

perc_dataset = 1.00
perc_valid = 0.20

X_train, Y_train, X_valid, Y_valid = preprocess_dataset(dataset, perc_dataset, perc_valid, load=reuse_dataset)
print("dataset: training (%d samples), validation (%d samples)" % (X_train.shape[0], X_valid.shape[0]))

# Neural Network Classifier
#----------------------------------------------------------------------------------------------

# NN parameters

epochs = 50
batch_size = 200
learning_rate = 0.001

filename = f"models/nn_{X_train.shape[0]}_{X_valid.shape[0]}_e{epochs}_bs{batch_size}_lr{learning_rate}_{epochs}"
im_filename = f"nn_{X_train.shape[0]}_{X_valid.shape[0]}_e{epochs}_bs{batch_size}_lr{learning_rate}.png"

retrain=True

# reshape into 28x28 features
X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid_reshaped = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

if not retrain and path.exists(filename):
    label = "loaded"
    model = NNClassifier(load=filename)
else:
    label = "trained"
    model = NNClassifier(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    history = model.train(X_train_reshaped, Y_train, X_valid_reshaped, Y_valid)
    print(history)

model.write(filename)

display_error_graph(
    model.error_data[:, 0], model.error_data[:, 1], model.error_data[:, 2],
    title="NN - Misclassification Error (Epochs = %s, Batch Size = %s, LR = %s)" % (epochs, batch_size, learning_rate),
    filename=im_filename
)
# %%
