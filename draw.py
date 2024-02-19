import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def draw(input_object, output, one_hot_labels, predicted_label):
    # Create a grid with 2 rows and 4 columns
    gs = gridspec.GridSpec(2, 4)

    # Create subplots
    ax1 = plt.subplot(gs[0, 0])  # Top left
    ax2 = plt.subplot(gs[1, 0])  # Bottom left
    ax3 = plt.subplot(gs[:, 1:])  # Second column


    ax1.imshow(input_object.original_image)
    ax1.set_title('Original Image')

    ax2.imshow(input_object.image_array, cmap='gray')
    ax2.set_title('Grayscale Image')

    ax3.bar(range(len(output)), output)
    ax3.set_xticks(range(len(output)))
    ax3.set_xticklabels(one_hot_labels.columns, rotation='vertical')
    ax3.set_title(predicted_label)

    # Show the figure
    plt.show()

def draw_neural_net(input_neurons, hidden_neurons, output_neurons):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for the input layer
    for i in range(input_neurons):
        G.add_node(f'Input {i+1}', subset='input')

    # Add nodes for the hidden layer
    for i in range(hidden_neurons):
        G.add_node(f'Hidden {i+1}', subset='hidden')

    # Add nodes for the output layer
    for i in range(output_neurons):
        G.add_node(f'Output {i+1}', subset='output')

    # Add edges between the input and hidden layers
    for i in range(input_neurons):
        for j in range(hidden_neurons):
            G.add_edge(f'Input {i+1}', f'Hidden {j+1}')

    # Add edges between the hidden and output layers
    for i in range(hidden_neurons):
        for j in range(output_neurons):
            G.add_edge(f'Hidden {i+1}', f'Output {j+1}')

    # Draw the graph
    pos = nx.multipartite_layout(G, subset_key='subset')
    nx.draw(G, pos, with_labels=True)
    plt.show()