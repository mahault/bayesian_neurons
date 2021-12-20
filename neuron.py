import numpy as np
from utils import get_likelihood_matrix, get_Abar, softmax, get_transition_matrix

class Neuron:
    """
    The Neuron is a simple active inference agent which is inferring the state of the network - 'firing' vs 'silent'.
    It does this based on binary observations Neuron.O which it receives from presynaptic neurons. These observations are received via the output_matrix.
    """

    def __init__(self, index, input_neuron=False):
        # each neuron has a unique index value which is used to for the connectivity and output matrices
        self.index = index

        # set elements of likelihood matricies
        self.init_precision = 0.9

        # Input neurons differ from hidden neurons in that they only receive a sinlge, user defined, input and do not update their precision belielfs.
        self.input_neuron = input_neuron

        # initialise inverse precision limits
        self.betaAm = np.zeros(2)
        self.betaAm = [0.5, 2.0]

        # define 'interspike interval'
        self.t_min = 1
        self.t_max = 3

    def connect(self, network):
        """
        Connect neuron to the rest of the network, given a network definition (and do everything else that depends
        on that).
        """
        self.connectivity_matrix = network.connectivity_matrix
        self.T = network.T
        
        # initialise beliefs
        self.X = np.zeros((2, self.T))
        self.X[:, 0] = [0.5, 0.5]
        self.Xbar = np.zeros((2, self.T))
        
        # use the connectivity matrix to find the number of pre-synaptic neurons
        # print("np sum self.connectivity_matrix",np.sum(self.connectivity_matrix))
        
        # u, counts = np.unique(self.connectivity_matrix[self.index], return_counts=True)
        # if self.input_neuron:
        #     self.number_of_presynaptic_neurons = 1
        # else:
        #     assert len(counts) == 2, "Neuron has no pre-synaptic connections. Check the connectivity matrix."
        #     self.number_of_presynaptic_neurons = counts[1]
        # print("self.index", self.index)
        if self.input_neuron == True:
            self.number_of_presynaptic_neurons = 1
        else:
            self.number_of_presynaptic_neurons = len(np.nonzero(network.connectivity_matrix[self.index])[0])
        # print("index", self.index)
        # print("number of presynaptic neurons",self.number_of_presynaptic_neurons)

        # TODO : Hook up layer's self.convolutions to the precision initialization
        ## for example if conv = [[1, 1], [0, 1]], set A matrix to be this for the neuron whose convolution this is
         
        # initialise likelihood matrices A and precision weighted likelihood matrices Abar
        self.A = get_likelihood_matrix(self.number_of_presynaptic_neurons, self.init_precision)
        # self.Abar = get_Abar(self.A, self.gammaA)
        
        # initialise sensory precision and inverse precision
        self.gammaA = np.ones((self.number_of_presynaptic_neurons, self.T))
        self.betaA = np.ones((self.number_of_presynaptic_neurons, self.T))

        # initialise observations, observation posteriors and outputs
        self.O = np.squeeze(np.zeros((self.number_of_presynaptic_neurons, self.T)))
        # print("inside connect number of presynaptic neurons", self.number_of_presynaptic_neurons)
        self.Obar = np.squeeze(np.zeros((self.number_of_presynaptic_neurons, 2, self.T)))
        self.O_out = np.zeros(self.T - 1)

    def perceive(self, output_matrix, connectivity_matrix, t):
        """
        This Neuron method receives the inputs for some time t, and calculates the posterior.
        """

        # take as observations the outputs of pre-synaptic neurons (for hidden neurons)
        if self.input_neuron == False:
            # find matrix of indexes of connnected pre-synaptic neurons
            connected_neurons = np.squeeze(np.where(connectivity_matrix[self.index]))
            for i, neuron_index in enumerate(connected_neurons):
                self.O[i][t] = output_matrix[neuron_index][t]

            # get precision weighted likelihood matrices for this timestep
            Abar = get_Abar(self.A, self.gammaA, t)

            # calculate perceputal evidence from all pre-synaptic inputs
            perceptual_evidence = np.zeros((2, len(Abar)))
            for input_index in range(len(Abar)):
                perceptual_evidence[:, input_index] = np.log(Abar[input_index, int(self.O[input_index, t]), :])
            tot_evidence = np.array((np.sum(perceptual_evidence[0]), np.sum(perceptual_evidence[1])))

            # calculate posterior belief
            # print("self.xbar",self.Xbar)
            # print("self.x",self.X)
            self.Xbar[:, t] = softmax(np.log(self.X[:, t]) + tot_evidence)

        # calculate posterior for input neuron (with single input)
        else:
            # print("self.xbar",self.Xbar)
            # print("self.x",self.X)
            # print("self.A",self.A)
            # print("self.O",self.O)
            # print("self.O shape",self.O.shape)
            self.Xbar[:, t] = softmax(np.log(self.X[:, t]) + np.log(self.A[0, int(self.O[t]), :]))

    def update_sensory_precision(self, t):
        """
        Caclulates the posterior beliefs about sensory precision.
        """

        # get observation posteriors (matrix) from discrete observations - used to calculated attentional charge
        for connection in range(self.number_of_presynaptic_neurons):
            # print("connection", connection)
            # print("number of presynaptic neurons",self.number_of_presynaptic_neurons)
            # print("shape obar", self.Obar.shape)
            self.Obar[connection, int(self.O[connection, t]), t] = 1

        # determine the precision weighted likelihood mapping for this time-step
        Abar = get_Abar(self.A, self.gammaA, t)

        # initialise the 'attentional charge' -  inverse precision updating term
        AtC = np.zeros(self.number_of_presynaptic_neurons)

        for s in range(self.number_of_presynaptic_neurons):  ## loop over synapses
            for i in range(2):  ## loop over outcomes
                for j in range(2):  ## loop over states
                    ### See "Uncertainty, epistemics and active inference" Parr, Friston.
                    AtC[s] += (self.Obar[s, i, t] - Abar[s, i, j]) * self.Xbar[j, t] * np.log(self.A[s, i, j])

            # limit size of update
            if AtC[s] > self.betaAm[0]:
                AtC[s] = self.betaAm[0] - 10 ** -5

            # create precision bounds
            self.betaA[s, t + 1] = self.betaA[s, t] - AtC[s]
            if self.betaA[s, t + 1] < self.betaAm[0]:
                self.betaA[s, t + 1] = self.betaAm[0]

            if self.betaA[s, t + 1] > self.betaAm[1]:
                self.betaA[s, t + 1] = self.betaAm[0]

            self.gammaA[s, t + 1] = self.betaA[s, t] ** -1

    def broadcast_beliefs(self, output_matrix, t):
        """
        Generates an 'action potential'. The decision to spike is sampled from the posterior probability distribution.
        """
        self.O_out[t] = np.random.choice([1, 0], p=self.Xbar[:, t])
        output_matrix[self.index][t] = self.O_out[t]

    def predict(self, t):
        """
        Calculate the prior beliefs for the next time-step.
        """
        B_t = get_transition_matrix(t, self.Xbar, self.t_min, self.t_max)
        self.X[:, t + 1] = np.inner(B_t, self.Xbar[:, t])
    