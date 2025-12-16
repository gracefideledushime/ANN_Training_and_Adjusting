import numpy as np

FEATURE_MIN = [0, 0, 0, 0]
FEATURE_MAX = [120, 10, 360, 400000]
FEATURE_COUNT = 4
LABELS = np.array([0, 1, 0, 1, 0, 1, 0])

data = [65, 5, 180, 80000], [120, 1, 72, 110000], [8, 6, 288, 50000], [50, 2, 324, 1600], [25, 9, 36, 160000], [80, 3, 120, 6000], [40, 3, 360, 400000]
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

scaled_data = scale_dataset(data, FEATURE_COUNT, FEATURE_MIN, FEATURE_MAX)

class NeuralNet:
    def __init__(self, features, labels, hidden_node_count):
        self.input = features
        # self.weights_input = np.array([[3.35, -5.82, 0.85, -4.35], [-3.94, 4.88, -1.83, 4.21], [2.43, -4.35, -0.36, -2.97], [-0.58, 1.30, 0.31, 1.40], [-3.34, 4.40, -0.75, 3.89]])
        self.weights_input = np.array([[3.35, -3.94, 2.43, -0.58, -3.34], [-5.82, 4.88, -4.35, 1.30, 4.40], [0.85, -1.83, 0.36, 0.31, -0.75], [-4.35, 4.21, -2.97, 1.40, 3.89]])
        self.hidden = np.zeros(hidden_node_count)
        self.weights_hidden = np.array([[8.45], [-6.96], [6.28], [-1.36], [-6.1]])
        self.expected_output = labels
        self.output = np.array([0] * len(labels))
    
    def forward_propagation(self):
        # calculate the sum of the weighted inputs
        print(f"Length of hidden node wum and output sum {len(self.weights_input)} and {len(self.input)}")
        hidden_weighted_sum = np.matmul(self.input, self.weights_input)
        print(self.input)
        print(f"Sum: \n{hidden_weighted_sum} \n")
        # calculate the sigmoid
        self.hidden = sigmoid(hidden_weighted_sum)
        print(f"Hidden node sum\n {self.hidden}")
        output_weighted_sum = np.matmul(self.hidden, self.weights_hidden)
        print(f"Before applying sigmoid to the output:\n {output_weighted_sum}\n")
        print(f"Output initially:\n {self.output}\n")
        self.output = sigmoid(output_weighted_sum)
    
        # print percentage output
        for i, example in enumerate(self.output):
            print(f"percentage of collision occurence for example {i + 1}: {round((example[0] * 100), 2)}%")
        print(f"\n-----------End of forward propagation----------\n")

# Train the Neural Net
def run_neural_net(epochs):
    nn = NeuralNet(scaled_data, LABELS, 5)
    for epoch in range(epochs):
        nn.forward_propagation()

run_neural_net(1)
