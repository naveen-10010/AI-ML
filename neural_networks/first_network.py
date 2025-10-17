import numpy as np

# Define Layer_Dense Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # Random weights
        self.bias = np.zeros((1, n_neurons))                        # Zero biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias      # Compute the output

# Sample Data
x = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

# Define and Use Layers
lay1 = Layer_Dense(4, 5)           # First layer with 4 inputs and 5 neurons
lay1.forward(x)                    # Forward pass for the first layer

lay2 = Layer_Dense(5, 2)           # Second layer with 5 inputs and 2 neurons
lay2.forward(lay1.output)          # Forward pass for the second layer

print("Output of layer 2:")
print(lay2.output)                 # Output of the second layer
