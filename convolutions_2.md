Convolutions 

SETUP NETWORK

- Feedforward threshold F

For each layer:
1. Define set of filters: diagonal left, diagonal right, all, none, etc...
2. If layer > 1: establish connections from previous layer

(Later, try removing priors from layers)

PREPROCESSING

1. Load in images
2. Turn into numpy arrays   
3. Break image into patches

PROCESSING

1. Choose an input image

2. Choose image patch
3. Compute activations for all neurons by comparing image patch to filter
4. Save state
5. Repeat from (2)

Once all patches have been processing:

1. Send all activities of most-activated neurons for each image patch [shaped as 2 x 2 x num_convolutions] to next layer
2. Break activities up into patches

Repeat steps (2)-(5) from PROCESSING above


Once all layers have been processed:

Activity of top-layer neuron = classification


## PSEUDOCODE

1. Define convolutions (as list or dict) - itertools?


2. Define Layer class (using Neuron class as element)


3. Define Network class (use Layers as components)
   a. Should take layer params as init arguments
   b. Auto-generate weight matrices ("connectivity matrices")
   c. Auto-generate graph of network
3. Build network with four layers (Input --> C1 --> C2 --> Output <-- Labels (Input 2)
    a. Instantiate Layer class for each layer
    b. Define connectivity of each layer/neuron (should be taken care of by Network class?)
   
4. Define generative process
    a. Input images and labels each cycle of the simulation   
      (Input to final layer should be coincident with feedforward signal)
    b. Similar to Lars's network
        i. Perceive
        ii. Precision update
        iii. Broadcast beliefs
        iv. Predict
   
5. Run it
