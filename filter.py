from convolutions import *
import math

class Filter():
    def __init__(self,image):
        self.pixels = int(math.sqrt(image["pixels"]))
        self.colors = image["colors"]
        self.convolutions = make_convolutions(self.pixels, self.colors)    
        self.filters= all_filters(int(math.sqrt(self.pixels)), self.colors)
        
        
image = {
    "pixels":2,
    "colors":[0]
}
new_filter = Filter(image)
# print(new_filter.convolutions)
# print(new_filter.filters)
# print(new_filter.filters.shape[0])