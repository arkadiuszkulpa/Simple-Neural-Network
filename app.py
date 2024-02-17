import os
import pandas as pd

from PIL import Image
import numpy as np

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neuralnetwork import NeuralNetwork

from draw import draw_neural_net

def extract_label_from_filename(filename):
    # Assuming the label is the first part of the filename before an underscore
    return filename.split('_')[0]

# Assuming your files are in a directory called 'shapes'
file_dir = 'Dataset'
filenames = os.listdir(file_dir)

# Extract labels from filenames
labels = [extract_label_from_filename(filename) for filename in filenames]

# Convert labels to a pandas Series
labels_series = pd.Series(labels)

# One-hot encode labels
one_hot_labels = pd.get_dummies(labels_series)
one_hot_labels['filename'] = filenames

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

# Open the file dialog and get the path of the selected file
file_path = filedialog.askopenfilename()

original_image = Image.open(file_path)

# Read and convert the image to grayscale
image = original_image.convert('L')  # 'L' mode is for grayscale

# Resize the image to 28x28 pixels
image = image.resize((28, 28))

# Convert the grayscale image to a NumPy array
image_array = np.array(image)

# Optionally, normalize the pixel values to be between 0 and 1
normalized_array = image_array / 255.0

# Flatten the image to create a 784-dimensional vector
input_vector = normalized_array.flatten()

# Now, input_vector can be used as the input to a neural network.

nn = NeuralNetwork(num_inputs=784, num_hidden=128, num_outputs=10)

print(input_vector)
output = nn.forward_propagation(input_vector)
print(output)






# Create a grid with 2 rows and 4 columns
gs = gridspec.GridSpec(2, 4)

# Create subplots
ax1 = plt.subplot(gs[0, 0])  # Top left
ax2 = plt.subplot(gs[1, 0])  # Bottom left
ax3 = plt.subplot(gs[:, 1:])  # Second column


ax1.imshow(original_image)
ax1.set_title('Original Image')

ax2.imshow(image_array, cmap='gray')
ax2.set_title('Grayscale Image')

ax3.bar(range(len(output)), output)
ax3.set_title('Output')

# Show the figure
plt.show()