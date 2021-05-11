
import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error, accuracy_score

class SVMClassifier:

    def __init__(self, regularization=1, epochs=-1, load=None):

        if load:
            # load from file
            self.clf = joblib.load(load)
        else:
            self.clf = SVC(
                C=regularization,
                max_iter=epochs,
                kernel="rbf",
                decision_function_shape="ovo"
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
        return 1 - accuracy_score(Y_true, self.clf.predict(X))

class NNClassifier:
    pass