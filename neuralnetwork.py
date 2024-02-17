import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

         # Initialize weights to small random values
        self.weights_1 = np.random.randn(num_hidden, num_inputs) * 0.01
        self.weights_2 = np.random.randn(num_outputs, num_hidden) * 0.01

        self.hidden_layer = [Neuron(self.weights_1[i], 0) for i in range(num_hidden)]
        self.output_layer = [Neuron(self.weights_2[i], 0) for i in range(num_outputs)]

    def forward_propagation(self, input_vector):
        hidden_layer_output = [np.dot(neuron.weights, input_vector) + neuron.bias for neuron in self.hidden_layer]
        output_layer_output = [np.dot(neuron.weights, hidden_layer_output) + neuron.bias for neuron in self.output_layer]
        return output_layer_output
    
    def train(self, input_vector, target_output, learning_rate):
        # Forward pass
        hidden_layer_output = [np.dot(neuron.weights, input_vector) + neuron.bias for neuron in self.hidden_layer]
        output_layer_output = [np.dot(neuron.weights, hidden_layer_output) + neuron.bias for neuron in self.output_layer]

        # Calculate error at output layer
        output_error = target_output - output_layer_output

        # Backpropagate error to hidden layer
        hidden_error = np.dot(output_error, self.weights_2.T)

        # Update weights in output layer
        for i, neuron in enumerate(self.output_layer):
            neuron.weights += learning_rate * output_error[i] * hidden_layer_output

        # Update weights in hidden layer
        for i, neuron in enumerate(self.hidden_layer):
            neuron.weights += learning_rate * hidden_error[i] * input_vector