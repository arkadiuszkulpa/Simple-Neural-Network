import os
import pandas as pd

import numpy as np

import tkinter as tk

from neuralnetwork import NeuralNetwork

from useractions import UserActions

from draw import draw

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

input1 = UserActions()


# Assuming your files are in a directory called 'shapes'
file_dir = 'Dataset'
filenames = os.listdir(file_dir)
filepaths = [os.path.join(file_dir, filename) for filename in filenames]
#https://data.mendeley.com/datasets/wzr2yv7r53/1

# Prepare the dataset
vectorize_or_load = tk.messagebox.askquestion('Vectorize or load dataset', 'Would you like to vectorize (yes) the dataset or load it (no)?')
print(vectorize_or_load)
if vectorize_or_load == 'yes':
    data = input1.prepare_dataset(filepaths)
    np.save('vectorized_dataset.npy', data)
    print('Data saved')
elif vectorize_or_load == 'no':
    data = np.load('vectorized_dataset.npy')
    print('Data loaded')

labels_series = input1.prepare_labels(filenames)

one_hot_labels = input1.prepare_one_hot_labels(labels_series)
#print(one_hot_labels)

nn = NeuralNetwork(num_inputs=784, num_hidden=128, num_outputs=9)
nnuntrained = NeuralNetwork(num_inputs=784, num_hidden=128, num_outputs=9)

train_or_test = tk.messagebox.askquestion('train or test', 'Would you like to train (yes) the dataset or test it (no)?')
if train_or_test == 'yes':
    nn.train_epochs(data, one_hot_labels, epochs=10, learning_rate=0.01)
elif train_or_test == 'no':
    nn.load_weights()
    nnuntrained.load_weights()
    print('Weights loaded')
#weights = input1.load_weights()

play = True
while play == True:
    input_vector = input1.produce_single_input()
    output = nn.forward_propagation(input_vector)

    # Find the index of the highest output value
    predicted_index = np.argmax(output)
    print(predicted_index)
    # Get the predicted label
    predicted_label = one_hot_labels.columns[predicted_index]
    print(predicted_label)
    draw(input1, output, one_hot_labels, predicted_label)
    #play = input1.ask_user_play_again()

root.destroy()