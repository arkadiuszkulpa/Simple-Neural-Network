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
        target_output = np.array(target_output)
        output_layer_output = np.array(output_layer_output)

        output_error = target_output - output_layer_output

        # Backpropagate error to hidden layer
        hidden_error = np.dot(output_error, self.weights_2)

        # Update weights in output layer
        for i, neuron in enumerate(self.output_layer):
            neuron.weights += learning_rate * np.array(output_error[i]) * np.array(hidden_layer_output)

        # Update weights in hidden layer
        for i, neuron in enumerate(self.hidden_layer):
            neuron.weights += learning_rate * hidden_error[i] * input_vector
        
        return output_layer_output

    def load_weights(self):
        """loads weights from a file"""
        weights = np.load('weights_9.npy', allow_pickle=True)
        self.weights_1 = weights.item().get('weights_1')
        self.weights_2 = weights.item().get('weights_2')

    def train_epochs(self, data, one_hot_labels, epochs, learning_rate):
        """trains the model the desired number of epochs"""
        model = self

        # Convert your DataFrame to a list of lists
        one_hot_labels_list = one_hot_labels.drop(columns='filename').values.tolist()

        # Now you can zip data and one_hot_labels_list together
        training_data = list(zip(data, one_hot_labels_list))

        for epoch in range(epochs):
            correct_predictions = 0
            print(f'Epoch {epoch + 1}/{epochs}')
            for input_vector, label in training_data:
                prediction = model.train(input_vector, label, learning_rate)
                if np.argmax(prediction) == np.argmax(label):
                    correct_predictions += 1
            accuracy = correct_predictions / len(training_data)
            #print(f'Accuracy: {accuracy * 100:.2f}%')
            print(f'Accuracy after epoch {epoch + 1}: {accuracy * 100}%')
            print(f'Epoch {epoch + 1}/{epochs} complete')

            # Gather weights from all layers into a dictionary
            weights = {
                'weights_1': model.weights_1,
                'weights_2': model.weights_2
            }

            # Save weights
            filename = f'weights_{epoch}.npy'
            np.save(filename, weights)