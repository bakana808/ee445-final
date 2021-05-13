
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

from models import SVMClassifier, NNClassifier

x = np.array(pd.read_csv("datasets/digit-recognizer/test.csv"))

nn_model = NNClassifier(load="models/nn_v1_20")
svm_model = SVMClassifier(load="models/svm_c100_100")

y_nn = nn_model.predict(x)
y_svm = svm_model.predict(x)



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

