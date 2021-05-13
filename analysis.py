# %%

import matplotlib.pyplot as plt
import numpy as np

from utils import display_error_graph, load_dataset
from models import SVMClassifier, NNClassifier


def display_confusion_matrix(model, title):

    _, _, x, y_true = load_dataset()

    y_pred = model.predict(x)
    misclasses = 0
    confusion_matrix = np.zeros((10, 10))

    for i in range(x.shape[0]):
        confusion_matrix[y_true[i], y_pred[i]] += 1
        if y_true[i] != y_pred[i]:
            misclasses += 1

    f = plt.figure(figsize=(10, 8))
    f.add_subplot(111)

    plt.imshow(np.log2(confusion_matrix + 1), cmap="YlGnBu")
    plt.colorbar()
    # plt.tick_params(size=5, color="white")
    plt.xticks(np.arange(0, 10), labels=np.arange(0, 10))
    plt.yticks(np.arange(0, 10), labels=np.arange(0, 10))

    threshold = confusion_matrix.max() / 2

    for i, j in np.ndindex((10, 10)):
        value = confusion_matrix[i, j]
        color = "white" if value > threshold else "black"
        plt.text(
            j,
            i,
            int(confusion_matrix[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            size=12,
            color=color,
        )

    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.title(title)
    plt.show()


print("SVM confusion matrix")
model = SVMClassifier(load="models/svm_c100_100")
display_confusion_matrix(model, "Confusion Matrix - SVM")

# %%

print("CNN confusion matrix")
model = NNClassifier(load="models/nn_v1_20")
display_confusion_matrix(model, "Confusion Matrix - CNN")


# %%
