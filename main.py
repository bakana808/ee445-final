"""
EE 445 - Handwritten Digit Recognition
Authors: Branden Akana, Alexa Fernandez
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %%

# load data
data = np.array(pd.read_csv("datasets/digit-recognizer/train.csv"))
tt_data = np.array(pd.read_csv("datasets/digit-recognizer/test.csv"))

rows = data.shape[0]  # total rows in training data
tr_rows = int(rows * 0.9)  # rows in training data
valid_rows = rows - tr_rows  # rows in validation data

# split training data into training and validation
tr_data = data[:tr_rows,:]
valid_data = data[tr_rows:,:]

print(f"training data:   { tr_data.shape }")
print(f"validation data: { valid_data.shape }")
print(f"test data:       { tt_data.shape }")

# display training data
f = plt.figure(figsize=(10, 10))

for i in range(25):
    label = str(tr_data[i, 0])
    img_data = tr_data[i, 1:].reshape((28, 28))  # image data, vals 0-255

    f.add_subplot(5, 5, i+1)
    plt.imshow(img_data, cmap="Greys", filternorm=False)
    plt.axis("off")
    plt.title(str(tr_data[i,0]))

plt.show()

# %%
