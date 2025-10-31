import numpy as np
import matplotlib.pyplot as plt

def cal_FR(data, start_time, end_time, bin_size_ms=1.0):
    """
    Calculates the firing rate (FR) in Hz for a binned spike train.

    The spike train is expected to be a 1D array of 0s and 1s (or False/True),
    where each index represents a single time bin.

    Args:
        data (np.ndarray or list): The binned spike train (e.g., [0, 1, 0, 0, ...]).
        start_time (int): The starting index (bin) of the calculation window.
                          This bin IS included in the calculation.
        end_time (int): The ending index (bin) of the calculation window.
                        This bin is NOT included (follows Python's slicing).
        bin_size_ms (float): The duration of a single time bin in milliseconds.
                             This is CRITICAL for converting counts to Hz.
                             Defaults to 1.0 (1 bin = 1 millisecond).

    Returns:
        float: The average firing rate in Hz (spikes per second) for the window.
    """
    
    # 1. Slice the data to get the window of interest
    # We use [start_time:end_time] which, in Python,
    # includes the 'start_time' index but excludes the 'end_time' index.
    # E.g., (2000, 2400) -> indices 2000, 2001, ..., 2399.
    spike_window = data[start_time:end_time]
    
    # 2. Count the spikes in that window
    # np.sum() is very fast and works on True/False or 1/0
    spike_count = np.sum(spike_window)
    
    # 3. Calculate the duration of the window in seconds
    num_bins = len(spike_window)
    
    if num_bins == 0:
        if start_time >= end_time:
            print(f"Warning: start_time ({start_time}) is >= end_time ({end_time}). Returning 0 Hz.")
        else:
            print("Warning: Data window is empty. Returning 0 Hz.")
        return 0.0
        
    duration_ms = num_bins * bin_size_ms
    duration_sec = duration_ms / 1000.0
    
    # 4. Calculate Firing Rate (Spikes / Second)
    firing_rate_hz = spike_count / duration_sec
    
    return firing_rate_hz

def plot_firing_rate_distribution(fr_array, title="Firing Rate Distribution"):
    """
    Plots a histogram of the firing rates from a 1D array.

    Args:
        fr_array (np.ndarray or list): A 1D array containing the firing rate
                                       for each neuron.
        title (str): The title for the plot.
    """
    
    if len(fr_array) == 0:
        print("Data array is empty, nothing to plot.")
        return

    # --- Create the Histogram ---
    
    # 'bins=30' is a good starting point. You can adjust this number.
    # 'edgecolor='k'' adds a black line to the edge of the bars,
    # which makes the plot look cleaner.
    plt.figure(figsize=(10, 6))
    plt.hist(fr_array, bins=30, edgecolor='k', alpha=0.7)
    
    # --- Add Labels and Title ---
    plt.title(title)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Number of Units (Count)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a vertical line for the mean firing rate
    mean_fr = np.mean(fr_array)
    plt.axvline(mean_fr, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {mean_fr:.2f} Hz')
    
    plt.legend()
    
    # --- Show the Plot ---
    # This will open a new window with your plot
    plt.show()

def plot_firing_rate_by_index(fr_array, title="Firing Rate by Trial Index"):
    """
    Plots the firing rate of each unit against its index in the array.
    The x-axis is the trial's index, and the y-axis is its firing rate.

    Args:
        fr_array (np.ndarray or list): A 1D array containing the firing rate
                                       for each neuron.
        title (str): The title for the plot.
    """
    
    if len(fr_array) == 0:
        print("Data array is empty, nothing to plot.")
        return

    # --- Create the X-axis (unit indices) ---
    unit_indices = np.arange(len(fr_array))

    # --- Create the Scatter Plot ---
    plt.figure(figsize=(10, 6))
    # A scatter plot is best here, as the index is just an
    # identifier, not a continuous variable.
    plt.scatter(unit_indices, fr_array, alpha=0.7, s=10) # s=10 for small dots
    
    # --- Add Labels and Title ---
    plt.title(title)
    plt.xlabel('Unit Index')
    plt.ylabel('Firing Rate (Hz)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for the mean firing rate
    mean_fr = np.mean(fr_array)
    plt.axhline(mean_fr, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {mean_fr:.2f} Hz')
    
    plt.legend()
    # Set x-limits to be clean
    plt.xlim(-1, len(fr_array))
    
    # --- Show the Plot ---
    plt.show()