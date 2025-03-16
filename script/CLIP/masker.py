import numpy as np

def extract_random_subsequence(x, seq_length, random_state=333):
    """
    Extract a random subsequence of specified length from a time series.
    
    Parameters:
    -----------
    x : numpy.array
        Input time series
    seq_length : int
        Length of subsequence to extract
    random_state : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    subsequence : numpy.array
        Extracted subsequence
    start_pos : int
        Starting position of the subsequence
    """
    
    np.random.seed(random_state)
    
    # Calculate valid starting positions
    max_start = len(x) - seq_length
    
    # Generate random starting position
    start_pos = np.random.randint(0, max_start + 1)
    
    # Extract subsequence
    subsequence = x[start_pos:start_pos + seq_length]
    
    # place subsequence at the begining of a sequence with the same length as x, fill the rest with nan
    subsequence = np.concatenate([subsequence.astype(float), np.nan * np.ones(len(x) - len(subsequence))])
    
    return subsequence, start_pos

# Example usage:
"""
# For a single time series
x = df_train.loc[0, ['1':'300']].values
subseq, start = extract_random_subsequence(x, seq_length=50)

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(x, 'b-', alpha=0.3, label='Original')
plt.plot(range(start, start + len(subseq)), subseq, 'r-', linewidth=2, label='Subsequence')
plt.grid(True)
plt.legend()
plt.show()
"""

def extract_n_random_subsequence(x, n, 
                                 min_length_ratio=1/6, # 300 * 1/6 = 50
                                 max_length_ratio=2/3, # 300 * 2/3 = 200
                                 step_size_ratio=1/30, # 300/6 = 50, 300/30 = 10
                                 seq_length=None, 
                                 random_state=333):
    """
    Extract n random subsequences of specified length from a time series.
    """
    subsequences = []
    start_positions = []
    n = min(n, 99) # no more than 99 subsequences
    step_size = round(len(x)*step_size_ratio)
    for i in range(n):
        # Move seq_length generation inside the loop
        current_seq_length = seq_length
        if current_seq_length is None:
            np.random.seed(random_state+i)
            # sample a new length for each subsequence
            current_seq_length = np.random.randint(int(len(x)*min_length_ratio), int(len(x)*max_length_ratio))
            # Round to nearest 50
            current_seq_length = round(current_seq_length/step_size) * step_size
            
        subseq, start = extract_random_subsequence(x, current_seq_length, random_state=random_state+i)
        subsequences.append(subseq.astype(float))
        start_positions.append(start)
        
        ## Print the length of each subsequence
        # print(f"Subsequence {i+1} length: {current_seq_length}")
        
    return subsequences, start_positions

# Example usage:
"""
subseq, start = extract_n_random_subsequence(x, n=5, seq_length=50, random_state=42)
print(subseq)
print(start)
# Visualize each subsequence
for i in range(len(subseq)):
    plt.figure(figsize=(12, 4))
    plt.plot(x, 'b-', alpha=0.3, label='Original')
    plt.plot(range(start[i], start[i] + len(subseq[i])), subseq[i], 'r-', linewidth=2, label='Subsequence')
    plt.grid(True)
    plt.legend()
plt.show()
"""




def plot_subsequences(x, subseq, start, n_cols=2):
    """
    Plot multiple subsequences in a grid layout.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    n_subseq = len(subseq)
    n_rows = (n_subseq + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle('Random Subsequences', fontsize=16)
    
    # Convert axes to 2D array if it's 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for i in range(n_subseq):
        # Get clean subsequence
        sub = subseq[i]
        sub = sub[~np.isnan(sub)]
        
        # Plot in corresponding subplot
        axes_flat[i].plot(x, 'b-', alpha=0.3, label='Original')
        axes_flat[i].plot(range(start[i], start[i] + len(sub)), 
                         sub, 'r-', linewidth=2, label='Subsequence')
        axes_flat[i].set_title(f'Subsequence {i+1}\nLength: {len(sub)}')
        axes_flat[i].grid(True)
        if i == 0:  # Only show legend for first subplot
            axes_flat[i].legend()
    
    # Hide empty subplots
    for i in range(n_subseq, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# # Example usage:
# subseq, start = extract_n_random_subsequence(x, n=10, random_state=333)
# plot_subsequences(x, subseq, start, n_cols=5)




def get_subseq_parallel(ts_df,
                        n, 
                       min_length_ratio=1/6,    # 300 * 1/6 = 50
                       max_length_ratio=2/3,    # 300 * 2/3 = 200
                       step_size_ratio=1/30,    # 300/30 = 10
                       seq_length=None, 
                       random_state=333):
    """
    Extract subsequences from multiple time series in parallel.
    
    Parameters:
    -----------
    ts_df : pandas.DataFrame
        DataFrame containing time series data
    n : int
        Number of subsequences per time series
    min_length_ratio : float
        Minimum subsequence length as ratio of time series length
    max_length_ratio : float
        Maximum subsequence length as ratio of time series length
    step_size_ratio : float
        Step size for rounding as ratio of time series length
    seq_length : int or None
        If specified, use fixed length for all subsequences
    random_state : int
        Random seed
    """
    from joblib import Parallel, delayed
    import multiprocessing
    import pandas as pd
    
    
    def process_row(x, id):

        str_id = str(id) # string versino of id for return df index
        if '_' in str_id:
            # split id by "_", keep the fist part (id of each raw time series), convert to int
            num_id = int(str_id.split('_')[0])
        else:
            num_id = int(str_id) # number version of id for random state
        """Process a single time series row"""
        subseq, _ = extract_n_random_subsequence(
            x=x, 
            n=n,
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            step_size_ratio=step_size_ratio,
            seq_length=seq_length,
            random_state=random_state + num_id
        )
        
        subseq_df = pd.DataFrame(subseq)
        subseq_df.columns = [str(i) for i in range(1, 301)]
        subseq_df['subid'] = range(1, subseq_df.shape[0]+1)
        subseq_df['raw_rowid'] = str_id
        return subseq_df
    
    # Determine number of cores
    try:
        total_cores = multiprocessing.cpu_count()
        n_cores = max(1, int(total_cores * 0.75))  # Use 75% of available cores
    except:
        n_cores = 4  # Default to 4 cores if can't determine
    
    print(f"Using {n_cores} cores for parallel processing")
    
    # Process all rows in parallel
    results = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(process_row)(row.values, idx) 
        for idx, row in ts_df.iterrows()
    )
    
    # Combine results
    combined_df = pd.concat(results, axis=0, ignore_index=True)

    # set raw_rowid as the index
    combined_df = combined_df.set_index('raw_rowid', inplace=False, drop=True).rename_axis(None)
    return combined_df


# Example usage:
"""
# Get time series columns

ts_cols = [str(i) for i in range(1, 301)]
ts_df = df_train[ts_cols].head(10)
combined_df = get_subseq_parallel(ts_df, n=5, step_size_ratio=1/30, random_state=333)

print(f"Resulting shape: {combined_df.shape}")


import matplotlib.pyplot as plt

# Get time series columns
ts_cols = [str(i) for i in range(1, 301)]

# Filter data for rowid 559
df_559 = df_train[df_train['rowid'] == 559]
# df_559 = df_test[df_test['rowid'] == 41332]

# Get unique subids
subids = df_559['subid'].unique()

# Create plot
plt.figure(figsize=(15, 8))

# Plot each subsequence
for subid in subids:
    subset = df_559[df_559['subid'] == subid]
    plt.plot(subset[ts_cols].values.T, label=f'subid={subid}', alpha=0.7)

    plt.title('Time Series for rowid=559 (All Subsequences)')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

"""