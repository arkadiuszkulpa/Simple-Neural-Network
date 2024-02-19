import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import pandas as pd

from functools import partial

class UserActions:
    def __init__(self):
        self.original_image = Image
        self.image_array = np.array
        self.weights = []

    def produce_single_input(self):
        """choose single picture and receive nn output"""
        # Open the file dialog and get the path of the selected file
        file_path = filedialog.askopenfilename()

        self.original_image = Image.open(file_path)

        # Flatten the image to create a 784-dimensional vector
        return self.prepare_data(file_path)

    def prepare_data(self, file_path):
        image = Image.open(file_path)
        # Read and convert the image to grayscale
        image = image.convert('L')  # 'L' mode is for grayscale

        # Resize the image to 28x28 pixels
        image = image.resize((28, 28))

        # Convert the grayscale image to a NumPy array
        image_array = np.array(image)

        # Optionally, normalize the pixel values to be between 0 and 1
        normalized_array = image_array / 255.0
        self.image_array = normalized_array
        # Flatten the image to create a 784-dimensional vector
        return normalized_array.flatten()

    def extract_label_from_filename(self, filename):
        # Assuming the label is the first part of the filename before an underscore
        return filename.split('_')[0]

    def prepare_dataset(self, file_paths):
        prepared_data = []
        for file in file_paths:
            prepared_data.append(self.prepare_data(file))
            print(f'Prepared data for {file}')
        return prepared_data
    
    def prepare_labels(self, filenames):
        # Extract labels from filenames
        labels = [self.extract_label_from_filename(filename) for filename in filenames]

        # Convert labels to a pandas Series
        return pd.Series(labels)

    def prepare_one_hot_labels(self, labels_series):
        # One-hot encode labels
        one_hot_labels = pd.get_dummies(labels_series)
        return one_hot_labels

    def load_weights(self):
        # Open the file dialog and get the path of the selected file
        file_path = filedialog.askopenfilename()
        self.weights = np.load(file_path, allow_pickle=True)
        return self.weights
