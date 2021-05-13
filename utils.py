"""
Functions that help with training and graphing/plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import joblib


def load_dataset():

    X_train = joblib.load("models/x_train")
    Y_train = joblib.load("models/y_train")
    X_valid = joblib.load("models/x_valid")
    Y_valid = joblib.load("models/y_valid")

    return X_train, Y_train, X_valid, Y_valid


def preprocess_dataset(dataset, perc_samples=1.0, perc_validation=0.1, load=False):
    """Reduces a dataset by a percentage and splits it into a training and validation dataset."""

    # load previously split datasets if possible
    try:
        if not load:
            raise Exception()

        return load_dataset()
    except Exception:
        pass

    dataset = np.copy(dataset)

    # shuffle row order
    np.random.shuffle(dataset)

    # reduce datasest
    rows = dataset.shape[0]  # total rows in dataset
    reduced_rows = int(rows * perc_samples)  # rows in reduced dataset
    dataset = dataset[:reduced_rows, :]

    # split training data into training and validation
    rows = dataset.shape[0]  # total rows in training data
    valid_rows = int(rows * perc_validation)  # rows in validation data
    tr_data = dataset[valid_rows:, :]
    valid_data = dataset[:valid_rows, :]

    # split dataset into inputs and outputs
    X_train = tr_data[:, 1:]
    Y_train = tr_data[:, 0].T

    X_valid = valid_data[:, 1:]
    Y_valid = valid_data[:, 0].T

    # save datasets
    joblib.dump(X_train, "models/x_train")
    joblib.dump(Y_train, "models/y_train")
    joblib.dump(X_valid, "models/x_valid")
    joblib.dump(Y_valid, "models/y_valid")

    return X_train, Y_train, X_valid, Y_valid


def display_samples(data: np.array):

    # display training data
    f = plt.figure(figsize=(10, 10))

    for i in range(25):
        label = str(data[i, 0])
        img_data = data[i, 1:].reshape((28, 28))  # image data, vals 0-255

        f.add_subplot(5, 5, i + 1)
        plt.imshow(img_data, cmap="Greys", filternorm=False)
        plt.axis("off")
        plt.title(str(data[i, 0]))

    plt.show()


def display_error_graph(epochs, train_errors, valid_errors, title: str, filename: str):

    f = plt.figure(figsize=(15, 6))  # figure

    plt.plot()
    plt.plot(epochs, train_errors, label="training error")  # accuracy curve
    plt.plot(
        epochs, valid_errors, label="validation error"
    )  # validation accuracy curve

    plt.title(title, fontsize=14)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Error %", fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.annotate(
        "%0.3f%%" % (train_errors[-1] * 100),
        xy=(1, train_errors[-1]),
        xytext=(8, -8),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
    )

    plt.annotate(
        "%0.3f%%" % (valid_errors[-1] * 100),
        xy=(1, valid_errors[-1]),
        xytext=(8, 8),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
    )

    plt.savefig(filename)
    plt.show()
