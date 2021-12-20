from layer import Layer
from convolutions import make_convolutions
from utils import *
from convolutions import *
from filter import Filter
import numpy as np
import math

class Network:
    '''
    Network class which takes the Layer class, and connects them in order. If the layer is input
    it creates it with a set of initial parameters connected to the number of convolutions present for the image. 
    
    '''
    
    def __init__(self, num_conv_layers, num_class_layers, T, num_labels, image):
        self.num_conv_layers = num_conv_layers
        self.num_class_layers = num_class_layers
        self.num_labels = num_labels
        self.filters = Filter(image).filters
        self.convolutions = Filter(image).convolutions
        self.layer_xs = np.array([4,2,2]) ###TODO: finish the layer_xs functions
        self.num_channels = [1,16,16] ###TODO:  finish the num channels function
        self.num_neurons, self.connectivity_matrix, self.neurons_per_group_per_layer = get_network_properties(self.layer_xs, self.num_channels, self.num_labels) #make sure to retun ints and test shape
        self.T = T
        self.output_matrix=np.zeros((len(self.connectivity_matrix),self.T))

        self.input_layer = Layer('input', None, 4, 0)
        self.conv_layers = []       
        for idx in range(num_conv_layers):
            self.conv_layers.append(Layer('convolutional', None, 2, idx+1))
        self.class_layers = []
        for idx in range(num_class_layers):
            self.class_layers.append(Layer('classification', None, 2, len(self.conv_layers)+idx+1))
        
        self.label_layer = Layer('label', None, self.num_labels, len(self.conv_layers)+idx+2)

        #input to conv1 layer
        
       #Get the total number of neurons for the entire network
        # total_neurons = input_layer.num_neurons
        # for conv in conv_layers :
        #     total_neurons += conv.num_neurons
        # for clas in class_layers :
        #     total_neurons += clas.num_neurons
        # #set a connectivity matrix with NP zeors for the number of neurons by the number of neurons
        # connectivity_matrix = np.zeros((total_neurons, total_neurons))
        
        # #Get all the indices of the neurons for all the layers
        # index_neurons_list = []
        # for neuron in input_layer.neurons :
        #     index_neurons_list += neuron.index
        # for layer in conv_layers:
        #     for neuron in layer:
        #         index_neurons_list += neuron.index
        # for layer in class_layers:
        #     for neuron in layer:
        #         index_neurons_list += neuron.index
        
        # #Get the image patches that we will be iterating over for the convolution filters   
        # num_patches = (input_layer.filter_size // conv_layers[0].filter_size )**2
        
        # #get the one coordinate as the beginning for each image patch   
        # beginning_coords = []
        # for i in itertools.islice(full_matrix[rows], rows_begiging, None, filter_size ):
        #         for j in itertools.islice(full_matrix[row], row_beginning, None, filter_size):
        #             beginning_coords.append([i,j])
        
        # #get all the coordinates in groups for each of the patches
        # patch_coordinates = []            
        # for image_patch in num_patches:
        #     for c in beginning_coords:
        #         coord_group=[]
        #         for i in range(filter_size):
        #             for j in range(filter_size):
        #                 ccol= c[0]+i
        #                 crow=c[1]+j
        #                 coord_group.append([ccol,crow])
        #     patch_coordinates.append(coord_group)
                          
                             
        # # isslice iterate over row number , over number of rows that corresponds to filter size
        # # then iterate over next batch of rows which corresponds to filter size number and do so again
        
            
        
        # for image_patch in num_patches:
        #     for filter in conv_layers[0].num_channels:
        
        # def coords_to_idx(coord_list, x_dix):
        #     return[x_dim*coords[1]+ coords[0]]
             
     