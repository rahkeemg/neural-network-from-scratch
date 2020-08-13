import numpy as np
import neural_network as nn
import plotly.express as px

np.random.seed(42)

# Set the inputs and weights using numpy random function
inputs = np.random.uniform(low=-10, high=10, size=4)
weights = np.random.uniform(low=-10, high=10, size=(3,4))
biases = np.random.randint( 0, 5, size=3)

print(f'Inputs: {inputs}')
print(f'Weights:\n {weights}')
print(f'Biases:\n {biases}')


layer_outputs = [] # Output of current layer
for n_weights, n_bias in zip(weights, biases):
    print(f"Output: {np.sum(n_weights * inputs) + n_bias}")

# print(f'Output of the network so far: {outputs}')
