# NEXT STEPS for convolution

### GOALS
1. Plot results of neural net test

   Test with exiting code:
    - 1. Try Lars's plot function
    - 2. Interpret / figure out what it means
    - 3. Adjust if needed 
    
    Steps:
    - 1. Spiking activity of each neuron over time
    - 2. Precision changes of weights over time
    - 3. Overall pattern of activity (visualize)
    
    Once we've completed through step (4) below:
    - 4. Plot overall accuracy and prediction VS truth per timestep
    

2. Add tests
      - inference
          - Test output of neurons
                - Given input image, select neurons in a given convolution, test whether
                    firing rate > threshold (e.g. 0.9)
   
      - learning
          - Test for large enough changes in precision 
      - classification
            - Test that only one output selected
            - Test that correct output is selected (error / correct rate) 
    

3. Feed in images
   - At start of perception loop, choose an image
   - Load image as array --> put in where image["values"] currently is

**4. Feed in labels
    - Correct get_network_properties to take label neurons into account in defining connectivity, etc.   
    - Define label layer in Network instance
    - Add label neuron activity calculation to perception loop
    - Load label (with image in 3.a) and set input to Input layer 2

5. Automate creating parameters: # layers, filter sizes(layer_xs)
    - Option 1: let filter size + input size determine no. of layers using a rule?
        - [set filter sizes]
    - Option 2: given filter size, image size, determine filter size given # layers?
        - [set num_layers]
    - Option 3: Combine these
    

6. Automate creating number of convolutions / channels per layers?
    - So far we've been assuming we want to use *all possible convolutions* of filter_size per layer
    - To use this rule:
        - use all_filters to get convolutions, use len of output to determine num_chanels (after input layer)
   
stretch goals:

7. allow custom stride:
    determine stride automatically as well given a rule?
    change get_network_properties to take stride as input
    automatically figure out zero-padding needed given stride + filter sizes
   
8. select convolutions using active inference
    a. define prior over convolutions
    b. determine which convolution to send activity to
    c. determine where in the image to send activity [3D location]
   
