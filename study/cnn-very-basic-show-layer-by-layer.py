import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

# Define input shape
input_shape = (5, 5, 1)  # Assuming grayscale images of size 28x28

# Define input tensor
inputs = Input(shape=input_shape)

# Create a Conv2D layer with two filters (3x3)
conv_layer = Conv2D(filters=2, kernel_size=(3, 3), activation='relu')(inputs)

# Build the model
model = Model(inputs=inputs, outputs=conv_layer)

# Generate random input data (example)
#input_data = np.random.rand(1, 5, 5, 1)
input_data = np.array([[[[1], [2], [3], [4], [5]],
                          [[1], [2], [3], [4], [5]],
                          [[1], [2], [3], [4], [5]],
                          [[1], [2], [3], [4], [5]],
                          [[1], [2], [3], [4], [5]]]])

# Print the input data
print("Input Data:")
print(input_data.squeeze())  # Squeeze is used to remove dimensions of size 1
print()

# Get the output of the convolutional layer
conv_output = model.predict(input_data)

# Modify the result of the convolutional layer (for demonstration purposes)
#modified_output = conv_output * 2

# Print the modified output
print("Modified Output of Convolutional Layer:")
print(conv_output.squeeze())

# Print the weights of the Conv2D layer
print("Weights of the Conv2D Layer:")
weights = model.layers[1].get_weights()[0]  # Get the weights (filter kernels)
biases = model.layers[1].get_weights()[1]  # Get the biases
print("Filter Weights:")
print(weights)
print("Biases:")
print(biases)