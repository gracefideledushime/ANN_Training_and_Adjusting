import numpy as np
import pandas as pd

FEATURE_MIN = [0, 0, 0, 0]
FEATURE_MAX = [120, 10, 360, 400000]
FEATURE_COUNT = 4
LABELS = []
data = []

# Drop the first column with the row number
file_data = pd.read_csv('car_collision.csv', usecols=range(1,6))
for i, row in file_data.iterrows():
    # print(f"{i} and {row['speed']}, {row['terrain_q']}, {row['deg_of_vision']}, {row['xp']}, {row['collision']}")

    # Add file data and labels to an array
    data.append([row['speed'], row['terrain_q'], row['deg_of_vision'], row['xp']])
    LABELS.append([0 if row['collision'] == "No" else 1])

# data = [65, 5, 180, 80000], [120, 1, 72, 110000], [8, 6, 288, 50000], [50, 2, 324, 1600], [25, 9, 36, 160000], [80, 3, 120, 6000], [40, 3, 360, 400000]
# each list is a node
# output_Weights =
# Convert all the data to a standard scale
def scale_dataset(dataset, ft_count, ft_min, ft_max):
    scaled_data = []
    for data in dataset:
        example = []
        for i in range(0, ft_count):
            example.append(scale_data_feature(data[i], ft_min[i], ft_max[i]))
        scaled_data.append(example)
    return scaled_data

# scale each value using its feature's min and max
def scale_data_feature(data, ft_min, ft_max):
    return round((data - ft_min) / (ft_max - ft_min), 3)

# activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# used in back propagation
def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

# transpose matrix to make n = p in (m,n) x (p,q) equal to make matrix multiplication possible
# def transpose(arr):
#     re_arranged_data = [[0]*len(arr) for _ in range(len(arr[0]))]

#     for i in range(len(arr)):
#         for j in range(len(arr[0])):
#             re_arranged_data[j][i] = arr[i][j]

#     return re_arranged_data

scaled_data = scale_dataset(data, FEATURE_COUNT, FEATURE_MIN, FEATURE_MAX)

class NeuralNetBackProp:
    def __init__(self, features, labels, hidden_node_count):
        self.input = features
        self.weights_input = np.array([[0.86, 0.17, 0.43, 0.27, 0.69], [0.22, 0.81, -0.02, 0.56, 0.81], [0.75, 0.19, 0.01, 0.77, 0.95], [0.70, 0.29, 0.76, 0.69, 0.38]])
        self.hidden = np.zeros(hidden_node_count)
        self.weights_hidden = np.array([[0.62], [0.43], [0.58], [0.70], [0.11]])
        self.weights_hidden_backprop = np.array([0.62, 0.43, 0.58, 0.70, 0.11])
        self.expected_output = labels
        self.output = np.array([0] * len(labels))

    def forward_propagation(self):
        # calculate the sum of the weighted inputs
        hidden_weighted_sum = np.matmul(self.input, self.weights_input)
        print(self.input)
        print(f"\nSum: \n{hidden_weighted_sum}")
        # calculate the sigmoid
        self.hidden = sigmoid(hidden_weighted_sum)
        print(f"\nHidden node sum\n {self.hidden}")
        output_weighted_sum = np.matmul(self.hidden, self.weights_hidden)
        self.output = sigmoid(output_weighted_sum)
        print(f"\n After applying sigmoid to output:\n {self.output}")

        # print percentage
        print("\nForward Propagation Output:\n")
        for i, example in enumerate(self.output):
            print(f"Percentage of collision occurence for example {i + 1}: {round((example[0] * 100), 2)}%")
        print("---------------- End of forward propagation --------------------")

    def back_propagation(self):
        cost = self.expected_output - self.output
        print(f"Expected output: \n{self.expected_output}\n")
        print(f"Output from forward propagation: \n{self.output}\n")
        print(f"\n Cost: \n{cost}\n")
        # print(f"types\n Output: {type(self.output)}\n Hidden node sum: {type(self.hidden)}")

        # Update weights between hidden nodes and output node
        weights_hidden_update = self.hidden * (2 * cost * sigmoid_deriv(self.output))
        print(f"Updating weight between hidden nodes and output node: {weights_hidden_update}\n")

        # print(f"test: \n {sigmoid_deriv(self.output)} \n hidden layer weights \n {(self.weights_hidden)}")
        # print(f"test: {sigmoid_deriv(self.output)} {len(sigmoid_deriv(self.output))}")
        # print(f"transpose hidden w {self.weights_hidden}")
        self.input = np.resize(self.input, (7,5))
        # print(f"transpose hidden w {type(self.input)} \n {self.input} \n{}")
        
        # Updates weights between input nodes and hidden nodes
        weights_input_update = self.input * (2 * cost * sigmoid_deriv(self.output) * np.resize(self.weights_hidden, (7,1))) * sigmoid_deriv(self.hidden)
        print(f"Updating weight between input nodes and hidden nodes: {weights_input_update}\n")
        
        self.weights_hidden = np.resize(self.weights_hidden, (7,5)) + weights_hidden_update
        print(f"New weights between hidden nodes and output node: \n{self.weights_hidden}\n")

        self.weights_input = np.resize(self.weights_input, (7,5)) + weights_input_update
        print(f"New weights between input nodes and hidden nodes: \n {self.weights_input}\n")
        print(f"---------------End of Part 2-------------")
# Train the Neural Net
def run_neural_net(epochs):
    nn = NeuralNetBackProp(scaled_data, LABELS, 5)
    for epoch in range(epochs):
        nn.forward_propagation()
        nn.back_propagation()

run_neural_net(1)
