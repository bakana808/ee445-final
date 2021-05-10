"""
EE 445 - Handwritten Digit Recognition
Authors: Branden Akana, Alexa Fernandez
"""
# %%
import pandas as pd
import numpy as np
import warnings

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error, accuracy_score

import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
warnings.warn = warn

def process_dataset(dataset, perc_samples = 1.0, perc_validation = 0.1):
    """Reduces a dataset by a percentage and splits it into a training and validation dataset."""

    dataset = np.copy(dataset)

    # randomize row order
    np.random.shuffle(dataset)

    # reduce datasest
    rows = dataset.shape[0]  # total rows in dataset
    reduced_rows = int(rows * perc_dataset)  # rows in reduced dataset
    dataset = dataset[:reduced_rows, :]

    # split training data into training and validation
    rows = dataset.shape[0]  # total rows in training data
    valid_rows = int(rows * perc_validation)  # rows in validation data
    tr_data = dataset[valid_rows:,:]
    valid_data = dataset[:valid_rows,:]

    x_train = tr_data[:, 1:]
    y_train = tr_data[:, 0].T

    x_valid = valid_data[:, 1:]
    y_valid = valid_data[:, 0].T

    return x_train, y_train, x_valid, y_valid

# load data
dataset = np.array(pd.read_csv("datasets/digit-recognizer/train.csv"))
tt_dataset = np.array(pd.read_csv("datasets/digit-recognizer/test.csv"))

# %%

def display_samples(data: np.array):

    # display training data
    f = plt.figure(figsize=(10, 10))

    for i in range(25):
        label = str(data[i, 0])
        img_data = data[i, 1:].reshape((28, 28))  # image data, vals 0-255

        f.add_subplot(5, 5, i+1)
        plt.imshow(img_data, cmap="Greys", filternorm=False)
        plt.axis("off")
        plt.title(str(data[i,0]))

    plt.show()

display_samples(tr_data)

# %%

#----------------------------------------------------------------------------------------------
# Support Vector Classifier
#----------------------------------------------------------------------------------------------

# parameters
num_epochs = [*range(50, 200, 10)]
regularization = 200
perc_dataset = 0.25
perc_valid = 0.10

X_train, Y_train, X_valid, Y_valid = process_dataset(dataset, perc_dataset, perc_valid)

print("%.2f%% training data (%d samples), %.2f%% validation (%d samples)" % (perc_dataset * 100, X_train.shape[0], perc_valid * 100, X_valid.shape[0]))

errors_train = []
errors_valid = []


for epochs in num_epochs:

    clf = SVC(
        C=regularization,
        max_iter=epochs,
        kernel="rbf",
        decision_function_shape="ovo"
    )

    clf.fit(X_train, Y_train)

    # error rate for training set
    error_train = 1 - accuracy_score(Y_train, clf.predict(X_train))

    # error rate for validation set
    error_valid = 1 - accuracy_score(Y_valid, clf.predict(X_valid))

    print("epochs -> %d \t train error -> %.06f \t valid error -> %.06f" % (epochs, error_train, error_valid))

    errors_train.append(error_train)
    errors_valid.append(error_valid)


f = plt.figure(figsize = (15,6)) #figure 

plt.plot()
plt.plot(num_epochs, errors_train, label = "training error") #accuracy curve 
plt.plot(num_epochs, errors_valid, label = "validation error") #validation accuracy curve

plt.title("SVM - Misclassification Error (regularization = %s)" % regularization, fontsize = 14)
plt.xlabel("Epochs", fontsize = 10)
plt.ylabel("Error %", fontsize = 10)
plt.grid(alpha = 0.3)
plt.legend()

plt.savefig('SVM_error.png')
plt.show()




# %%
