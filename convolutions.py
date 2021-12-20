import numpy as np
from itertools import combinations_with_replacement as cb

def make_convolutions(pixels,colors):
    return list(cb(colors, pixels))

def all_filters(pixels, values=[0,255]):
    """
    Returns a matrix containing all possible filters of size (`pixels` x `pixels`) and with
    high and low value specified in `values`.

    Partial explanation:

    N = pixels*pixels  # number of pixels in square (= number of binary digits in corresponding array)
    num_filters = 1<<N  # use bit shift to get 2^N, the number of binary strings of length N
    base_numbers = np.arange(num_filters, dtype='uint8')[:, None]  # list the numbers in the range 2^N
    shifts = np.arange(N, dtype='uint8')[::-1]  # get reversed list of
    one = 0b1  # binary representation of the number 1
    filters = base_numbers >> shifts & one  # shift the numbers 0-2^N by `shifts`, then use bitwise AND which is equivalent to %2
    """

    # Get base filters
    N = pixels*pixels
    filters = np.arange(1<<N)[:,None] >> np.arange(N)[::-1] & 0b1

    # Convert to desired values
    filters *= max(values)
    filters[np.where(filters==0)] = min(values)
    
    print("filters", filters)
    print("filters shape", filters.shape)

    return filters

#testing
# make_convolutions(4,[255, 0])
# all_filters(2, [0, 255])