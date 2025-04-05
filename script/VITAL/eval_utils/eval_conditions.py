import numpy as np

def successive_increases(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive increases
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for increase, 0 for decrease/same)
    y_diff = np.diff(y)
    y_bin = (y_diff > 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single increases (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of increases
    out['d'] = np.mean(y_bin == 0)  # proportion of decreases/same
    
    # Two consecutive (length 2)
    n2 = len(y_bin) - 1
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # up,up
    out['ud'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 0))  # up,down
    out['du'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 1))  # down,up
    out['dd'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 0))  # down,down
    
    # Three consecutive (length 3)
    n3 = len(y_bin) - 2
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    # Calculate entropy for each length
    def calc_entropy(p):
        # Remove zeros and calculate entropy
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    # Entropy calculations
    out['h'] = calc_entropy(np.array([out['u'], out['d']]))  # entropy of length 1
    out['h2'] = calc_entropy(np.array([out['uu'], out['ud'], out['du'], out['dd']]))  # entropy of length 2
    
    return out

# Example usage:
# y = [1, 2, 1, 3, 2, 4, 5]
# result = successive_increases(y)

def successive_unchanges(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive unchanges
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for no change, 0 otherwise)
    y_diff = np.diff(y)
    y_bin = (y_diff == 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single unchanges (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of unchanges
    
    # Two consecutive unchanges (length 2)
    n2 = len(y_bin) - 1
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # unchange,unchange
    
    # Three consecutive (length 3)
    n3 = len(y_bin) - 2
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    return out

# Example usage:
# y = [1, 1, 1, 2, 2, 3, 3, 3]
# result = successive_unchanges(y)