
from utils import *

from neuron import Neuron

from convolutions import all_filters
import math
import numpy as np

class Layer:
    """
    This class instantiates the layers of the neural net.
    Layers will be connected through the Network class.
    Layers are composed of neurons that instantiate the Neuron class from utils
    param `filter_size` determines size of convolutional filter for conv layers or input size for input layers
    """

    def __init__(self, layer_type='convolutional', convolutions=None, filter_size=2, layer_num=0):
        self.layer_type = layer_type
        self.is_input_layer = self.layer_type in ['input', 'label']
        self.layer_num = layer_num
        self.filter_size = filter_size
        self.convolutions = convolutions

    def connect(self, network):
        """
        Connect to the rest of the network, once a network has been specified
        """
        self.connectivity_matrix = network.connectivity_matrix
        self.T = network.T

        # Set up convolutional layer
        if self.layer_type == 'convolutional':
            # Define convolutions (or use pre-defined set passed to __init__)
            if self.convolutions is None:
                self.convolutions = all_filters(self.filter_size)

            self.num_neurons = network.neurons_per_group_per_layer[self.layer_num]
            self.num_channels = len(self.convolutions)
            # print("self.num_neurons", self.num_neurons)
            # print("self.num_channels", self.num_channels)

        # Set up input layer
        elif self.layer_type == 'input':
            self.num_neurons = self.filter_size**2
            self.num_channels = 1  # Assume 1 color channel for now

        elif self.layer_type == 'classification' or 'label':
            self.num_neurons = self.filter_size
            self.num_channels = 1

        # Add neurons that make a feature map for each convolutional filter (or color channel, for input layers)
        self.neurons = self.add_neurons(network)
        #self.neurons = [self.add_neuron_group(network) for x in range(self.num_channels)]
        # print("self.neurons", self.neurons)

        for neuron_group in self.neurons:
            for neuron in neuron_group:
                neuron.connect(network)
                # print("neuron.index",neuron.index)

    def calculate_num_neurons(self, network):
        """
        Calculate number of neurons needed to cover input layer, given stride, input layer size, filter size
        For now assume a stride of the same size as the filter, and no padding.
        """
        num_input_x, num_input_y = network.layer_dims[self.layer_num - 1]
        assert num_input_x % self.filter_size == 0, "Please choose a filter size that evenly divides the input"
        assert num_input_y % self.filter_size == 0, "Please choose a filter size that evenly divides the input"
        num_neurons_x = num_input_x // self.filter_size
        num_neurons_y = num_input_y // self.filter_size

        return num_neurons_x * num_neurons_y

    def perception_loop(self, network, t):
        self.perceive(network, t)
        if t < network.T-1:
            self.broadcast_beliefs(network, t)
            self.predict(t)
            if not self.is_input_layer:
                self.update_sensory_precision(t)

    def perceive(self, network, t):
        for neuron_group in self.neurons:
            for neuron in neuron_group:
                # print("neuron", neuron.index)
                neuron.perceive(network.output_matrix, network.connectivity_matrix, t)
    def broadcast_beliefs(self, network, t):
        for neuron_group in self.neurons:
            for neuron in neuron_group:
                neuron.broadcast_beliefs(network.output_matrix, t)
    def predict(self,t):
        for neuron_group in self.neurons:
            for neuron in neuron_group:
                neuron.predict(t)
    def update_sensory_precision(self, t):
        for neuron_group in self.neurons:
            for neuron in neuron_group:
                neuron.update_sensory_precision(t)   
    
    @staticmethod
    def add_neuron(idx, is_input_neuron, network):
        return Neuron(
            idx,
            is_input_neuron
        )

    def add_neurons(self, network):
        neurons = []
        for channel_idx in range(self.num_channels):
            neurons.append(
                [
                    self.add_neuron(
                        int(np.sum(network.num_neurons[:self.layer_num])) + idx + (channel_idx*self.num_neurons),
                        self.is_input_layer,
                        network
                    ) for idx in range(self.num_neurons)
                ])

        return neurons
        # return [
        #     self.add_neuron(
        #         int(np.sum(network.num_neurons[:self.layer_num])) + idx, self.is_input_layer, network
        #     ) for idx in range(self.num_neurons)
        # ]
    

# network = {
#     "num_neurons" : 0,
#     "connectivity_matrix": get_connectivity_matrix(int(math.sqrt(16)), int(math.sqrt(4)), 16), #make sure to retun ints and test shape
#     "T":2,
#     "layer_dims":3,
# }

