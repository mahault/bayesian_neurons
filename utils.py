import numpy as np
import itertools
from convolutions import *

def softmax(X):                                                                 ###converts log probabilities to probabilities
  norm = np.sum(np.exp(X)+10**-5)
  Y = (np.exp(X)+10**-5)/norm
  return Y

def softmax_dim2(X):                                                            ###converts matrix of log probabilities to matrix of probabilities
  norm = np.sum(np.exp(X)+10**-5,axis=0)
  Y = (np.exp(X)+10**-5)/norm
  return Y

def normalise(X):                                                               ###normalises a matrix of probabilities
  X= X/np.sum(X,0)
  return X


def get_likelihood_matrix(number_of_presynaptic_neurons, initial_confidence):
  ''' 
  Generates the likelihood matrices given the number of inputs and initial observation precision. 
  '''
  p = initial_confidence
  assert 0 <= p <= 1, "Confidence must be in range [0,1]"
  A = np.zeros((number_of_presynaptic_neurons,2,2))
  for i in range(number_of_presynaptic_neurons):
    A[i,:,0] = [1-p , p] #State indexed by 0 = firing? -- so then p(firing | observe not firing) = 1 - precision
    A[i,:,1] = [p , 1-p]
  return A

def get_Abar(A, gammaA, t): ######## debug - need Abar for each timestep?
    ''' 
    Generates precision weighted likelihood matrices given the matrix of likelihood matricies and a precision vector.  
    '''
    assert len(A) == len(gammaA), "Number of matricies and precisions must match."
    Abar = np.zeros((len(A),2,2))
    for i in range(len(A)):
        Abar[i,:,:] = softmax_dim2(np.log(A[i])*gammaA[i,t])
    return Abar


def get_transition_matrix(t, Xbar, t_min, t_max):
    '''
    Inspired by Palacios et al. 2019
    The tranisition matrix depends on the neuron's 'memory' of it's own recent activity. 
    If the neuron has fired in the 'interspike interval' then it expects to continue firing. 
    This slightly unusual formulation is re-evaluated every timestep and approximates a deep temporal model. 
    '''
    recent_fire = 0
    B = np.zeros((2,2))
    for x in Xbar[0,max(0,t-t_max):max(0,t-t_min)]:
        if x > 0.5: 
            recent_fire = 1
            break
    if recent_fire == 0:
        B[:,0]=[.2,.8]
        B[:,1]=[.2,.8]
    else:
        B[:,0]=[.8,.2]
        B[:,1]=[.8,.2]  
    return B
    

class OldNeuron:
    """
    The Neuron is a simple active inference agent which is inferring the state of the network - 'firing' vs 'silent'.
    It does this based on binary observations Neuron.O which it receives from presynaptic neurons. These observations are received via the output_matrix. 
    
    """
    def __init__(self, index, connectivity_matrix, T, input_neuron = False):
        # each neuron has a unique index value which is used to for the connectivity and output matrices
        self.index = index

        # set elements of likelihood matricies
        self.init_precision = 0.9

        # Input neurons differ from hidden neurons in that they only receive a sinlge, user defined, input and do not update their precision belielfs. 
        self.input_neuron = input_neuron

        # use the connectivity matrix to find the number of pre-synaptic neurons
        u, counts = np.unique(connectivity_matrix[self.index], return_counts=True)
        if input_neuron:
            self.number_of_presynaptic_neurons = 1
        else:
            assert len(counts) == 2, "Neuron has no pre-synaptic connections. Check the connectivity matrix."
            self.number_of_presynaptic_neurons = counts[1]
        
        # initialise sensory precision and inverse precision
        self.gammaA = np.ones((self.number_of_presynaptic_neurons,T))
        self.betaA = np.ones((self.number_of_presynaptic_neurons,T))

        # initialise inverse precision limits
        self.betaAm = np.zeros(2)
        self.betaAm = [0.5,2.0]

        # initialise likelihood matrices A and precision weighted likelihood matrices Abar
        self.A = get_likelihood_matrix(self.number_of_presynaptic_neurons, self.init_precision)
        # self.Abar = get_Abar(self.A, self.gammaA)

        # initialise beliefs
        self.X = np.zeros((2,T))
        self.X[:,0] = [0.5,0.5]
        self.Xbar = np.zeros((2,T))
        
        # initialise observations, observation posteriors and outputs
        self.O = np.squeeze(np.zeros((self.number_of_presynaptic_neurons,T)))
        self.Obar = np.squeeze(np.zeros((self.number_of_presynaptic_neurons,2,T)))
        self.O_out = np.zeros(T-1)

        # define 'interspike interval'
        self.t_min = 1
        self.t_max = 3

    def perceive(self, output_matrix, connectivity_matrix, t):
        """
        This Neuron method receives the inputs for some time t, and calculates the posterior.
        """

        # take as observations the outputs of pre-synaptic neurons (for hidden neurons)
        if self.input_neuron==False:
            # find matrix of indexes of connnected pre-synaptic neurons
            connected_neurons = np.squeeze(np.where(connectivity_matrix[self.index]))
            for i, neuron_index in enumerate(connected_neurons):
                self.O[i][t] = output_matrix[neuron_index][t]
        
            # get precision weighted likelihood matrices for this timestep
            Abar = get_Abar(self.A, self.gammaA, t)

            # calculate perceputal evidence from all pre-synaptic inputs
            perceptual_evidence = np.zeros((2,len(Abar)))
            for input_index in range(len(Abar)):
                perceptual_evidence[:,input_index] = np.log(Abar[input_index,int(self.O[input_index,t]),:])
            tot_evidence = np.array((np.sum(perceptual_evidence[0]),np.sum(perceptual_evidence[1])))

            # calculate posterior belief
            self.Xbar[:,t] = softmax(np.log(self.X[:,t])+tot_evidence)

        # calculate posterior for input neuron (with single input)
        else:
            self.Xbar[:,t] = softmax(np.log(self.X[:,t])+np.log(self.A[0,int(self.O[t]),:]))

    def update_sensory_precision(self, t):
        """
        Caclulates the posterior beliefs about sensory precision.
        """

        # get observation posteriors (matrix) from discrete observations - used to calculated attentional charge
        for connection in range(self.number_of_presynaptic_neurons):
            self.Obar[connection, int(self.O[connection,t]),t]=1

        # determine the precision weighted likelihood mapping for this time-step
        Abar = get_Abar(self.A, self.gammaA, t)

        # initialise the 'attentional charge' -  inverse precision updating term
        AtC = np.zeros(self.number_of_presynaptic_neurons)

        for s in range(self.number_of_presynaptic_neurons):     ## loop over synapses
            for i in range(2):                                  ## loop over outcomes
                for j in range(2):                              ## loop over states
                ### See "Uncertainty, epistemics and active inference" Parr, Friston.
                    AtC[s] += (self.Obar[s,i,t]-Abar[s,i,j])*self.Xbar[j,t]*np.log(self.A[s,i,j])

            # limit size of update
            if AtC[s] > self.betaAm[0]:
                AtC[s] = self.betaAm[0]-10**-5

            # create precision bounds
            self.betaA[s,t+1] = self.betaA[s,t] - AtC[s]
            if self.betaA[s,t+1] < self.betaAm[0]:
                self.betaA[s,t+1] = self.betaAm[0]
            
            if self.betaA[s,t+1] > self.betaAm[1]:
                self.betaA[s,t+1] = self.betaAm[0]

            self.gammaA[s,t+1] = self.betaA[s,t]**-1


    def broadcast_beliefs(self, output_matrix, t):
        """
        Generates an 'action potential'. The decision to spike is sampled from the posterior probability distribution.
        """
        self.O_out[t] = np.random.choice([1,0], p=self.Xbar[:,t])
        output_matrix[self.index][t] = self.O_out[t]

    def predict(self, t):
        """
        Calculate the prior beliefs for the next time-step.
        """
        B_t = get_transition_matrix(t, self.Xbar, self.t_min, self.t_max)
        self.X[:,t+1] = np.inner(B_t, self.Xbar[:,t])


## Convolution utils

def input_indices(x_dim, fs, input_channels, initial_index=0):
    """
    Compact function to get groups of indices in input to hook up to each neuron in a convolutional layer
    """
    num_patches = (x_dim // fs) ** 2
    filter_x = np.arange(0, fs)

    base_block = np.array(
        [np.array([filter_x, filter_x+x_dim]) + channel_idx * x_dim**2 for channel_idx in range(input_channels)]
    )

    return np.array(
        [base_block + x_dim*((idx*fs)//x_dim) + idx*fs for idx in range(num_patches)]
    ) + initial_index


def get_connectivity_matrix(x_dim, filter_size, num_conv_channels):
    """
    Return connectivity matrix, given:
    `x_dim`: x (=y) dimension of (square) input image
    `filter_size`: x (=y) dimension of (square) convolutional filter
    `num_conv_channels`: number of filters in convolutional layer
    """

    # TODO: See if this can be generalized to build the connectivity matrix layer by layer
    # TODO : Could probably do something smart with binary arrays
    # TODO : Pass a list of filters instead of one filter size
    # TODO : Figure out how to pass the actual number of layers
    # TODO : Make sure you can get the value of layer_xs from the list of filters
    ii = input_indices(x_dim, filter_size, num_conv_channels)
    last_used_idx = ii[-1, -1, -1]
    num_filter_neurons = (x_dim // filter_size) ** 2
    num_neurons = x_dim**2 + num_filter_neurons*num_conv_channels
    connectivity_matrix = np.zeros((num_neurons, num_neurons), dtype='int')
    for idx, patch in enumerate(ii):
        for channel in range(num_conv_channels):
            connectivity_matrix[last_used_idx + idx + num_filter_neurons*channel, patch] = 1

    return connectivity_matrix


def get_network_properties(layer_xs, num_channels, num_labels=0):
    """
    Return [num_neurons(x) for x in num_layers], connectivity matrix, given:
    `layer_x_dims`: [x (=y) dimension of (square) filter per layer (for input layer, filter = whole image)]
    `num_channels`: [number of channels (# of color channels for input, # of filters for conv) per layer]
    """

    #TODO: FIX BUG: neurons in last layer only see 1 of the conv channels in previous layer (in conn. matrix)
    #TODO: Factor in classification/label layer!!
    #TODO: Default option to generate convolutions and figure out num_channels automatically (except for input layer)

    input_size = layer_xs[0]**2
    neurons_per_channel = [input_size]
    total_layer_neurons = input_size * num_channels[0]
    num_neurons = [total_layer_neurons]

    iis = []
    neurons_per_layer = []
    total_neurons = 0
    for layer_idx in range(1, len(layer_xs)):
        iis.append(input_indices(layer_xs[layer_idx-1], layer_xs[layer_idx], num_channels[layer_idx-1], total_neurons))
        total_neurons += total_layer_neurons
        num_layer_neurons = (layer_xs[layer_idx-1] // layer_xs[layer_idx]) ** 2
        neurons_per_channel.append(num_layer_neurons)
        total_layer_neurons = num_layer_neurons*num_channels[layer_idx]
        num_neurons.append(total_layer_neurons)

    total_neurons += total_layer_neurons + (num_labels*2)

    print("iis")
    for idx, ii in enumerate(iis):
        print(idx, ii)

    connectivity_matrix = np.zeros((total_neurons, total_neurons), dtype='int')
    for layer_idx in range(1, len(layer_xs)):
        for idx, patch in enumerate(iis[layer_idx-1]):
            for channel in range(num_channels[layer_idx]):
                print("layer", layer_idx)
                print("patch", idx)
                print("channel", channel)
                print("n", num_neurons[layer_idx-1] + idx + neurons_per_channel[layer_idx]*channel)
                connectivity_matrix[np.sum(num_neurons[:layer_idx]) + idx + neurons_per_channel[layer_idx]*channel, patch] = 1

    if num_labels:
        neurons_so_far = total_neurons - num_labels*2
        for patch_idx, patch in enumerate(iis[len(layer_xs)-2]):
            print("patch_idx", patch_idx)
        for idx in range(num_labels):
            connectivity_matrix[neurons_so_far + idx, list(range(neurons_so_far-num_neurons[-1], neurons_so_far))] = 1
            connectivity_matrix[neurons_so_far + idx, list(range(total_neurons-num_labels, total_neurons))] = 1

        neurons_per_channel += [num_labels, num_labels]
        num_neurons += [num_labels, num_labels]

    # [input, conv1, conv2, ..., classification, labels]
    return num_neurons, connectivity_matrix, neurons_per_channel


def connect_to_conv(connectivity_matrix, input_indices, num_conv_channels):
    """
    Given array of [input image indices] for each image patch and number of convolutions in conv layer,
    connect `connectivity_matrix` by placing 1s in appropriate entries
    """
    filter_size = input_indices[0].shape[0]**2
    
    last_used_idx = input_indices[-1, -1, -1]
    for idx, patch in enumerate(input_indices):
        for channel in range(num_conv_channels):
            connectivity_matrix[patch, last_used_idx + idx + filter_size*channel] = 1

    return connectivity_matrix


#### Testing
# layer_xs = all_filters(2,[1,2])
# layer_xs = np.array([4,2,2])
# print(layer_xs.shape )
# num_channels = [2,2,2]
# get_network_properties(layer_xs,num_channels)
