import numpy as np
import matplotlib.pyplot as plt
from utils import Neuron
from layer import Layer
from network import Network
import networkx as nx

# initialise parameters
# timesteps
T = 20
number_of_neurons = []

# define connections between rows of post-synaptic neurons and columns of pre-synaptic neurons (1 if connected)
connectivity_matrix = np.zeros((number_of_neurons, number_of_neurons))

# connecting 2 input neurons to hidden neuron
connectivity_matrix[2] = np.array([1, 1, 0])

# define matrix of neuron spikes
output_matrix = np.zeros((len(connectivity_matrix), T))

# %%
# initialise neurons

# input neurons
input_neuron_0 = Neuron(0, connectivity_matrix, T, input_neuron=True)
input_neuron_1 = Neuron(1, connectivity_matrix, T, input_neuron=True)

# hidden neurons
neuron_2 = Neuron(2, connectivity_matrix, T)

# %%
# give inputs to input neurons
input_neuron_0.O[1] = 1
input_neuron_0.O[2] = 1
input_neuron_0.O[3] = 1
input_neuron_0.O[4] = 1
input_neuron_0.O[5] = 1

input_neuron_1.O[4] = 1
input_neuron_1.O[5] = 1
input_neuron_1.O[6] = 1
input_neuron_1.O[7] = 1
input_neuron_1.O[8] = 1

# input_neuron_0.O[:] = 1
# input_neuron_1.O[:] = 1

# %%
# run message passing
# Each neuron does four steps for each moment. 
# 1. Perception and belief updating about the state of the network
# 2. Update beliefs about sensory precision (ie. synaptic weighting update)
# 3. Broadcast beliefs about the network state through action (potentials).
# 4. Predict the next timestep (forming next priors) 

for t in range(T):
    # input to pre-synaptic input neurons
    input_neuron_0.perceive(output_matrix, connectivity_matrix, t)
    input_neuron_1.perceive(output_matrix, connectivity_matrix, t)

    if t < T - 1:
        input_neuron_0.broadcast_beliefs(output_matrix, t)
        input_neuron_1.broadcast_beliefs(output_matrix, t)

        input_neuron_0.predict(t)
        input_neuron_1.predict(t)

    # # post-synaptic neuron
    neuron_2.perceive(output_matrix, connectivity_matrix, t)

    if t < T - 1:
        neuron_2.update_sensory_precision(t)
        neuron_2.broadcast_beliefs(output_matrix, t)
        neuron_2.predict(t)

# %%
# plotting results

spike_times_0 = np.where(output_matrix[0])
spike_times_0 = spike_times_0[0]

spike_times_1 = np.where(output_matrix[1])
spike_times_1 = spike_times_1[0]

spike_times_2 = np.where(output_matrix[2])
spike_times_2 = spike_times_2[0]

plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.title('Posterior beliefs of input neuron 0')
plt.plot(np.arange(0, T), input_neuron_0.Xbar[0, :], label=r'N_0 ${\bar{X}}}$', color='coral')
plt.scatter(np.arange(T), input_neuron_0.O, label='N0 inputs', color='orangered')
for spike in spike_times_0:
    plt.axvline(x=spike, label='N_0 Spikes', color='coral', linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')
plt.ylim([0, 1.0])
plt.yticks([-0.05, 1.05], ['silent', 'firing'])
plt.xlim([0, T])

plt.subplot(2, 2, 3)
plt.title('Posterior beliefs of input neuron 1')
plt.plot(np.arange(0, T), input_neuron_1.Xbar[0, :], label=r'N_1 ${\bar{X}}}$', color='green')
plt.scatter(np.arange(T), input_neuron_1.O, label='N1 inputs', color='green')
for spike in spike_times_1:
    plt.axvline(x=spike, label='N_1 Spikes', color='green', linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')
plt.ylim([0, 1.0])
plt.yticks([-0.05, 1.05], ['silent', 'firing'])
plt.xlim([0, T])

plt.subplot(2, 2, 2)
plt.title('Posterior beliefs of hidden neuron 2')
plt.plot(np.arange(0, T), neuron_2.Xbar[0, :], label=r'N_2 ${\bar{X}}}$', color='blue')
plt.scatter(np.arange(T), neuron_2.O[0], label='N2 inputs', color='blue')
plt.scatter(np.arange(T), neuron_2.O[1], label='N2 inputs', color='blue')
for spike in spike_times_2:
    plt.axvline(x=spike, label='N_2 Spikes', color='blue', linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')
plt.ylim([0, 1.0])
plt.yticks([-0.05, 1.05], ['silent', 'firing'])
plt.xlim([0, T])

plt.subplot(2, 2, 4)
plt.title('Precision on observations')
plt.plot(np.arange(0, T), neuron_2.gammaA[0, :], label='gammaA[0]', color='coral')
plt.plot(np.arange(0, T), neuron_2.gammaA[1, :], label='gammaA[1]', color='green')
plt.xlim([0, T])
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
# %%
