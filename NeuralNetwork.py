import imp
import random
import math
from arabic_dataset import *
# imp.reload(arabic_dataset)

# Node for a neural netowrk
# It's set up so you can traverse the network up or down (may be overkill)
class Node:

    def __init__(self):
        self.activation = 0.0  # current activation of the node
        self.inputs = []  # list of input nodes--empty for input layer
        self.weights = []  # list of float for weights--empty for output layer
        self.outputs = []  # list of output nodes--empty for output layer
        self.weight_change = 0.0 # variable created for backprop

    # for printing a node
    def __str__(self):
        output = ""
        output += "There are " + str(len(self.inputs)) + " input nodes.\n"
        output += "There are " + str(len(self.outputs)) + " output nodes.\n"
        output += "Weights are " + str(self.weights) + "\n"
        output += "Change in weights are " + str(self.weight_change) + "\n"
        output += "Activation is " + str(self.activation) + "\n"
        return output

    # Computes the linear combination. It can either apply logistic 
    # or the nondifferentialbe if > 0, 1 else 0
    def update_activation(self):
        total = 0.0
        for i, input_node in enumerate(self.inputs):
            total += input_node.activation * self.weights[i]

        # apply nonlinearity
        # The simple version
        # self.activation = 0 if total < 0 else 1
        # logistic function as describe on wikipedia for neural networks. Set beta = 10
        self.activation = 1.0 / (1.0 + math.pow(math.e,-2*10*total))



# Class for the network
class Network:

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):

        # Initialize our three variables:
        self.input_layer = [ Node() for i in range(input_layer_size)]
        self.hidden_layer = [ Node() for i in range(hidden_layer_size)]
        self.output_layer = [ Node() for i in range(output_layer_size)]

    # to print out a network ( a more complete version of this is commented out below)
    def __str__(self):
        output = ""
        output += "===============OUTPUT LAYER===============\n"
        for node in self.output_layer:
            output += str(node)

        return output


    """
    def __str__(self):
        output = ""
        output += "There are " + str(len(self.input_layer)) + " nodes in the input layer.\n"
        output += "There are " + str(len(self.hidden_layer)) + " nodes in the hidden layer.\n"
        output += "There are " + str(len(self.output_layer)) + " nodes in the output layer.\n"
        output += "===============INPUT LAYER===============\n"
        for node in self.input_layer:
            output += str(node)
        output += "===============HIDDEN LAYER===============\n"
        for node in self.hidden_layer:
            output += str(node)
        output += "===============OUTPUT LAYER===============\n"
        for node in self.output_layer:
            output += str(node)

        return output
    """

    # Adds all the connections to the network after it has been initialized.
    def connect_all(self):

        # Connect input layer to hidden layer
        for input_node in self.input_layer:
            for hidden_node in self.hidden_layer:
                input_node.outputs.append(hidden_node)
        for hidden_node in self.hidden_layer:
            for input_node in self.input_layer:
                hidden_node.inputs.append(input_node)
                hidden_node.weights.append(random.uniform(-1,1))

        # Connect hidden layer to output layer
        for hidden_node in self.hidden_layer:
            for output_node in self.output_layer:
                hidden_node.outputs.append(output_node)
        for output_node in self.output_layer:
            for hidden_node in self.hidden_layer:
                output_node.inputs.append(hidden_node)
                output_node.weights.append(random.uniform(-1,1))

    # Evaluates the whole network given some input values
    def evaluate(self, input):
        # set the input values
        for i, input_node in enumerate(self.input_layer):
            input_node.activation = float(input[i])
        # update activation in the hidden layer
        for i, hidden_node in enumerate(self.hidden_layer):
            hidden_node.update_activation()
        # update activation in the output layer
        for i, output_node in enumerate(self.output_layer):
            output_node.update_activation()

    # Just returns the current output state of the network
    def get_output(self):
        output = []
        for output_node in self.output_layer:
            output.append(output_node.activation)
        return output

    # This is the learning rule function for weights between input and hidden layer. 
    # It takes in the coefficients from the GA and updates the weights.
    def update_hidden_layer_weights(self, coeffs):

        for input_node in self.input_layer:
            input_node.weight_change = 0.0

        for j,hidden_node in enumerate(self.hidden_layer):
            for i in range(len(hidden_node.weights)):

                original_weight = hidden_node.weights[i] 

                weight_change = \
                    coeffs[1] * hidden_node.weights[i] + \
                    coeffs[2] * hidden_node.activation + \
                    coeffs[4] * hidden_node.inputs[i].activation + \
                    coeffs[5] * hidden_node.weights[i] * hidden_node.weights[i] + \
                    coeffs[6] * hidden_node.weights[i] * hidden_node.activation + \
                    coeffs[8] * hidden_node.weights[i] * hidden_node.inputs[i].activation + \
                    coeffs[9] * hidden_node.activation * hidden_node.activation + \
                    coeffs[11] * hidden_node.activation * hidden_node.inputs[i].activation + \
                    coeffs[14] * hidden_node.inputs[i].activation * hidden_node.inputs[i].activation + \
                    coeffs[15] * hidden_node.weight_change + \
                    coeffs[16] * hidden_node.weight_change * hidden_node.weight_change + \
                    coeffs[17] * hidden_node.weight_change * hidden_node.inputs[i].activation + \
                    coeffs[18] * hidden_node.weights[i] * hidden_node.weight_change + \
                    coeffs[19] * hidden_node.activation * hidden_node.weight_change


                # scale the output
                weight_change *= coeffs[0]
                print "Delta weight: " + str(weight_change)
                hidden_node.weights[i] += weight_change

                # make sure it's within bounds
                if hidden_node.weights[i] > 1: hidden_node.weights[i] = 1
                if hidden_node.weights[i] < -1: hidden_node.weights[i] = -1

                # This is storing the weight changes for backprop. 
                # Haven't confirmed it's working exactly yet.
                hidden_node.inputs[i].weight_change += abs(hidden_node.weights[i] - original_weight)


    # This is the learning rule function for weights between hidden and output layer. 
    # It takes in the coefficients from the GA and updates the weights.
    # It's slightly different from the one above since it has access to different variables.
    def update_output_layer_weights(self, target, coeffs):

        for hidden_node in self.hidden_layer:
            hidden_node.weight_change = 0.0

        for j,output_node in enumerate(self.output_layer):
            for i in range(len(output_node.weights)):

                original_weight = output_node.weights[i] 

                weight_change = \
                    coeffs[1] * output_node.weights[i] + \
                    coeffs[2] * output_node.activation + \
                    coeffs[3] * target[j] + \
                    coeffs[4] * output_node.inputs[i].activation + \
                    coeffs[5] * output_node.weights[i] * output_node.weights[i] + \
                    coeffs[6] * output_node.weights[i] * output_node.activation + \
                    coeffs[7] * output_node.weights[i] * target[j] + \
                    coeffs[8] * output_node.weights[i] * output_node.inputs[i].activation + \
                    coeffs[9] * output_node.activation * output_node.activation + \
                    coeffs[10] * output_node.activation * target[j] + \
                    coeffs[11] * output_node.activation * output_node.inputs[i].activation + \
                    coeffs[12] * target[j] * target[j] + \
                    coeffs[13] * target[j] * output_node.inputs[i].activation + \
                    coeffs[14] * output_node.inputs[i].activation * output_node.inputs[i].activation 

                # scale the output
                weight_change *= coeffs[0]
                # print "Delta weight: " + str(weight_change)
                output_node.weights[i] += weight_change

                # make sure it's within bounds
                if output_node.weights[i] > 1: output_node.weights[i] = 1
                if output_node.weights[i] < -1: output_node.weights[i] = -1

                output_node.inputs[i].weight_change += abs(output_node.weights[i] - original_weight)

