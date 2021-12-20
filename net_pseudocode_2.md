# Finish connectivity matrix + layer setup function

1. Generalize `get_connectivity_matrix` to use all layers
2. And to return number of neurons per layers
3. Also figure out how to return number of convolutions per layer (or pass as param?)
4. (Figure out how to) hook up second input layer (for classification)
4.5 Determine which precisions to use

# Use this info [number of neurons per layer, number of channels per  layer, connectivity_matrix] to set up network

5. Fill in arguments to Layer-creation code using outputs from `get_connectivity_matrix`
5. 5 Make a filter class

6. Make sure that the way indices are assigned to neurons when Layers are created is consistent with the convention used to build the connectivity matrix
    - Print out / verify how indices are assigned in setting up connectivity_matrix
    - Look at hwo indices are assigned in Layer-creation loop in Layer class
    - Change one (probably the latter) if needed

6.5. Define "activate" or "fire" or "perceive" method for Layer (+ possibly Network?) + "broadcast" methods
    - Layer level:
        - Define the new function(s)
        - Look at Lars's code and figure out how to feed inputs into a series of neurons (in a Layer) as a loop / (array?)
        - Insert code into the function that makes the Layer's neurons fire accordingly

    - Network level:
        - Define "input" or "fire" or "perceive" function
            - This should input image to input layer and label to label layer
            - And just define the gen. process (Layers firing one after another)

    # Set something up at the Network level that defines the gen. process and propagates info through Neurons using their methods
    # Layers could (should?) be used to define the temporal order of firing for neurons

7. Define generative process (see 6.5  -Network level)
    a. Draw inputs + labels from the space of possible inputs each timestep (random)
    b. Activate layer 1 (input)
    c. Activate layer 2 (conv1)
    d. Activate layer 3 (conv2)
    e. Activate layer 4 (label/input2)
    e. Activate layer 5 (classifier) using input from (d) and (e)

8. Modify Neuron/Layer class:
    a. Take precision as input param (for each weight for each neuron)
    b. Make this depend on layer type --> If "classifier" type, raise relative precision of input layer
