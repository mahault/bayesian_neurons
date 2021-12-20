#get the right imports
from convolutions import *
from filter import *
from layer import *
from neuron import *
import os
from network import Network
from utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.name == 'posix':
    mpl.use('tkagg')
import random
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import cv2 as cv
import glob

#set up the network


# layer_xs = np.array([4,2,2])
# print(layer_xs.shape )
# num_channels = [1,16,16]
# get_network_properties(layer_xs,num_channels)
num_conv_layers = 1
num_class_layers = 1
T= 700
num_labels = 1



### Pass in a real image
files_train = glob.glob("train/*")
files_test = glob.glob("test/*")
print(files_train)

train_images = []
test_images = []
for img in files_train:
    train_images.append([cv.imread(img),img[8]])
    print("train_images",train_images)
for img in files_test:
    test_images.append([cv.imread(img),img[8]])
    print("test_images",test_images)

test_pixels = []
for img in test_images:
    temp = []
    for i in range(img[0].shape[0]):
        for j in range(img[0].shape[1]):
            temp.append(list(img[0][i,j]))
    test_pixels.append([temp,img[1]])
    print("temp", [temp,img[1]])

for idx, img in enumerate(test_pixels):
    print("img", img)
    print("idx", idx)
    for idxp, px in enumerate(img[0]):
        print("px",px)
        if px[0] < 50:
            
            test_pixels[idx][0][idxp] = 0
        else:
            test_pixels[idx][0][idxp] = 1
        if test_pixels[idx][1] == "x":
            test_pixels[idx][1] = [1,0]
        else:
            test_pixels[idx][1] = [0,1]
    print("image", img)  
    
    
    
    
train_pixels = []
for img in train_images:
    temp = []
    for i in range(img[0].shape[0]):
        for j in range(img[0].shape[1]):
            temp.append(list(img[0][i,j]))
    train_pixels.append([temp,img[1]])
    print("temp", [temp,img[1]])

for idx, img in enumerate(train_pixels):
    print("img", img)
    print("idx", idx)
    for idxp, px in enumerate(img[0]):
        print("px",px)
        if px[0] < 50:
            
            train_pixels[idx][0][idxp] = 0
        else:
            train_pixels[idx][0][idxp] = 1
        if train_pixels[idx][1] == "x":
            train_pixels[idx][1] = 1
        else:
            train_pixels[idx][1] = 0
    print("image", img)   


###Begin Sequence 
  
image = {
    "pixels":len(train_pixels[0][0]),
    "colors":[2],
    
}
print("image.pixels", image["pixels"])
network = Network( num_conv_layers, num_class_layers, T, num_labels, image)

# print(network)
# print(dir(network))
# print(dir(network.input_layer))
#instantiate the layers


#instantiate the neurons and #connect the structure
network.input_layer.connect(network)
for conv_layer in network.conv_layers:
    
    conv_layer.connect(network)
for class_layer in network.class_layers:
    class_layer.connect(network)


    # for row in network.connectivity_matrix:
    #     print(row)


    #define perception loop

for img in train_pixels: 
    image = {
    "pixels":len(train_pixels[0][0]),
    "colors":[2],
    "values":img[0],
    "label": img[1]
    }
    
    for t in range(network.T):
        # Input layer
        for neuron_group in network.input_layer.neurons:
            for neuron in neuron_group:
                # print(neuron.O[t])
                neuron.O[t]=image["values"][neuron.index]
        
        # print("t", t)
        network.input_layer.perception_loop(network, t)
        for conv_layer in network.conv_layers:
            conv_layer.perception_loop(network,t)

        # TODO : Add label layer

        for class_layer in network.class_layers:
            class_layer.perception_loop(network, t)
        

    #pass data to network

    #plot results

#### TEST

accuracy = 0

for img in test_pixels: 
    image = {
    "pixels":len(test_pixels[0][0]),
    "colors":[2],
    "values":img[0],
    "label": img[1]
    }
    
    for t in range(network.T):
        for neuron_group in network.input_layer.neurons:
            for neuron in neuron_group:
                # print(neuron.O[t])
                neuron.O[t]=image["values"][neuron.index]
        
        # print("t", t)
        network.input_layer.perception_loop(network, t)
        for conv_layer in network.conv_layers:
            conv_layer.perception_loop(network,t)
        for class_layer in network.class_layers:
            class_layer.perception_loop(network, t)
        

    #pass data to network

    #plot results




    layers = []
    layers.append(network.input_layer)
    for layer in network.conv_layers:
        layers.append(layer)
    for layer in network.class_layers:
        layers.append(layer)
    
    # print(layers)


    spike_times=[]
    for row in network.output_matrix:
        spike_times.append(np.where(row))

    # print("spike_times list", spike_times)
    fig = plt.figure(figsize=(30,30))




            
    print("STARTING")
    for layer in layers:
        # print(f"LAYER {layer.layer_num}")
        # print(f"NUM GROUPS {len(layer.neurons)}")
        for idx, neuron_group in enumerate(layer.neurons):
            # print(f"GROUP {idx}")
            for neuron in neuron_group:
                # print("neuron index", neuron.index)
    
                # print("all dark conv")
    
                color = (random.random(), random.random(), random.random())
                # fig.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
                # fig.scatter(np.arange(T),neuron.O,label=neuron.index,color='orangered')
                # print("np.arrange.shape input",np.arange(0,T).shape)
                # print("neuron.Xbar[0,:].shape input",neuron.O.shape)
                if np.arange(0,T).shape == neuron.O.shape:
                    plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)
                    plt.scatter(np.arange(T-1),neuron.O_out,label=neuron.index,color=color)
                else:
                    plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)
                    plt.scatter(np.arange(T-1),neuron.O_out,label=neuron.index,color=color)
                # plt.plot(np.arange(0,T),network.input_layer.neurons[0][0].Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
                # plt.scatter(np.arange(T),network.input_layer.neurons[0][0].O,label='N0 inputs',color='orangered')
                # print("spike times neuron index",spike_times[neuron.index][0])
                for spike in spike_times[neuron.index][0]:
                    plt.axvline(x=spike,  label = 'N Spikes',color='coral', linestyle='--')



    # # plt.title('Posterior beliefs of input neuron 0')
    # for neuron_group in network.input_layer.neurons:
    #     for neuron in neuron_group:
    #         color = (random.random(), random.random(), random.random())
    #         # fig.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    #         # fig.scatter(np.arange(T),neuron.O,label=neuron.index,color='orangered')
    #         print("np.arrange.shape input",np.arange(0,T).shape)
    #         print("neuron.Xbar[0,:].shape input",neuron.O.shape)
    #         print("O_out.shape",neuron.O_out.shape)

    #         # Posterior beliefs
    #         #plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)

    #         # Firing / not firing
    #         #plt.scatter(np.arange(T - 1),neuron.O_out,label=neuron.index,color=color)

    #         # Sensory precision
    #         plt.plot(np.arange(0, T), neuron.gammaA[0], '-.', label=r'{neuron.index} ${\bar{gammaA}}}$', color=color)

    # # plt.plot(np.arange(0,T),network.input_layer.neurons[0][0].Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    # # plt.scatter(np.arange(T),network.input_layer.neurons[0][0].O,label='N0 inputs',color='orangered')
    #         print("spike times neuron index",spike_times[neuron.index][0])
    #         for spike in spike_times[neuron.index][0]:
    #             plt.axvline(x=spike,  label = 'N Spikes',color='coral', linestyle='--')
    
    # for conv_layer in network.conv_layers[:1]:
    #     print("conv_layer", conv_layer)
    #     for neuron_group in conv_layer.neurons:
    #         print("neuron_group",neuron_group)
    #         for neuron in neuron_group:
    #             print("neuron", neuron)
    #             color = (random.random(), random.random(), random.random())
    #             # fig.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    #             # fig.scatter(np.arange(T),neuron.O,label=neuron.index,color='orangered')
    #             print("np.arrange.shape conv",np.arange(0,T).shape)
    #             print("neuron.Xbar[0,:].shape conv",neuron.O[1].shape)
    #             print("neuron.Xbar[0,:].shape conv",neuron.O_out[1].shape)
    #             plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)
    #             plt.scatter(np.arange(T-1),neuron.O_out,label=neuron.index,color=color)
    #             # plt.plot(np.arange(0,T),network.input_layer.neurons[0][0].Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    #             # plt.scatter(np.arange(T),network.input_layer.neurons[0][0].O,label='N0 inputs',color='orangered')
    #             print("spike times neuron index",spike_times[neuron.index][0])
    #             for spike in spike_times[neuron.index][0]:
    #                 plt.axvline(x=spike,  label = 'N Spikes',color='coral', linestyle='--')


    # for class_layer in network.class_layers:
    #     for neuron_group in class_layer.neurons:
    #         for neuron in neuron_group:
    #             color = (random.random(), random.random(), random.random())
    #             # fig.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    #             # fig.scatter(np.arange(T),neuron.O,label=neuron.index,color='orangered')
    #             print("np.arrange.shape class",np.arange(0,T).shape)
    #             print("neuron.Xbar[0,:].shape class",neuron.Xbar[0,:].shape)
    #             plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)
    #             plt.scatter(np.arange(T),neuron.O[1],label=neuron.index,color=color)
    #             # plt.plot(np.arange(0,T),network.input_layer.neurons[0][0].Xbar[0,:],label=r'N_0 ${\bar{X}}}$',color='coral')
    #             # plt.scatter(np.arange(T),network.input_layer.neurons[0][0].O,label='N0 inputs',color='orangered')
    #             print("spike times neuron index",spike_times[neuron.index][0])
    #             for spike in spike_times[neuron.index][0]:  
    #                 plt.axvline(x=spike,  label = 'N Spikes',color='coral', linestyle='--')
        
    
   
    for neuron_group in class_layer.neurons:
            print("neuron_group",neuron_group)
            avgn0 = 0
            avgn1 = 0
            for neuron in neuron_group:
                print("neuron class", neuron)
                color = (random.random(), random.random(), random.random())
                plt.plot(np.arange(0,T),neuron.Xbar[0,:],label=r'{neuron.index} ${\bar{X}}}$',color=color)
                plt.scatter(np.arange(T-1),neuron.O_out,label=neuron.index,color=color)
                
                ##CLASSIFY 
                #TODO: compare with actual label to see success rate
                print("image label", image["label"])
                if neuron_group.index(neuron) == 1 :
                    avgn0 = np.average(neuron.O_out[-20:])
                    print("avg", avgn1)
                else :
                    avgn1 = np.average(neuron.O_out[-20:])
                    print("avg", avgn0)
                # if neuron_group.index(neuron) == 1 and avg > 0.5:
                #     print("this image is X", neuron.O_out[-1] )
                    
                #     if image["label"] == [0,1]:
                #         accuracy += 1
                # elif neuron_group.index(neuron) == 0 and avg > 0.5:
                #     print("this image is O", neuron.O_out[-1])
                #     if image["label"] == [1,0]:
                #         accuracy += 1
                print("spike times neuron index",spike_times[neuron.index][0])
                for spike in spike_times[neuron.index][0]:
                    plt.axvline(x=spike,  label = 'N Spikes',color='coral', linestyle='--')
            if avgn1 > avgn0:
                print("this image is X", neuron.O_out[-1] )
                    
                if image["label"] == [1,0]:
                    accuracy += 1
            elif avgn1 < avgn0:
                print("this image is O", neuron.O_out[-1])
                if image["label"] == [0,1]:
                    accuracy += 1
            else : 
                print("unable to determine")
                
    for spike in spike_times:
        for el in list(spike[0]):
        # print(list(spike[0]))
            plt.axvline(x=el,  label = 'N Spikes',color='coral', linestyle='--')
            # print("did a thing")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 'best')
    plt.ylim([0,1.0])
    plt.yticks([-0.05,1.05],['silent','firing'])
    plt.xlim([0,T])

    
# plt.show() 
plt.savefig('neurons.jpg')   
print("accuracy",accuracy/len(test_images))