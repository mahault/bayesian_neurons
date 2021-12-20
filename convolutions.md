Convolutions
pseudocode!

SETUP NETWORK

- Function to compute similarities
- Function to set connections based on similarities
- Similarity threshold S
- Feedforward threshold F

For each layer:
1. Define set of filters: diagonal left, diagonal right, all, none, etc...
2. Compute similarities between filters 
3. Establish lateral connectivity based on [filters, S]
4. If layer > 1: establish connections from previous layer

(Later, try removing priors from layers)

PREPROCESSING

1. Load in images
2. Turn into numpy arrays   
3. Break image into patches

PROCESSING

1. Choose an input image

2. Choose image patch
3. Pick a neuron in layer 1 randomly
4. Compute match with image patch -> Activity
5. Check if activity of any neuron > F
6.  If so: 
      - Save state
    Else:
      - Send activity through lateral connections (except those to neurons from which activity has been received already at t)
7. Repeat from (5)

Once all patches have been processing:

1. Send all activities of neurons [shaped as 2 x 2 x num_convolutions] to next layer
2. Break activities up into patches

Repeat steps (2)-(7) from PROCESSING above


Once all layers have been processed:

Activity of top-layer neuron = classification
