import numpy as np
import pandas as pd

def get_correct_trials(data, session_name='session_00'):
    """
    Filters a DataFrame for a specific session, validates 'trialNum',
    re-indexes the data to be 0-based (trialNum - 1), and returns
    the new index of all "correct" trials (where correct == 1).

    Args:
        data (pd.DataFrame): The main DataFrame containing all trial data.
                             Must include 'session', 'trialNum', and 'correct'
                             columns.
        session_name (str): The name of the session to filter for.

    Returns:
        pd.Index: A pandas Index (e.g., [0, 1, 3, ...]) corresponding to
                  the 0-indexed trial number (trialNum - 1) for
                  all correct trials.
    """
    
    # --- 1. Filter by session ---
    # Selects only the rows that match the given session_name
    session_data = data[data['session'] == session_name]
    
    if session_data.empty:
        print(f"Warning: No data found for session '{session_name}'.")
        return pd.Index([]) # Return an empty index

    # --- 2. Sort by 'trialNum' ---
    # This is a crucial step to ensure the data is in the correct order.
    sorted_data = session_data.sort_values(by='trialNum')
    
    # --- 3. NEW: Validate 'trialNum' ---
    trial_nums = sorted_data['trialNum'].values
    
    if len(trial_nums) > 0:
        # Check if it starts at 1
        if trial_nums[0] != 1:
            print(f"Warning for session '{session_name}': "
                  f"'trialNum' starts at {trial_nums[0]}, not 1.")

        # Check for gaps (if there's more than one trial)
        if len(trial_nums) > 1:
            diffs = np.diff(trial_nums)
            if not np.all(diffs == 1):
                print(f"Warning for session '{session_name}': "
                      f"'trialNum' column is not contiguous (has gaps or duplicates).")
    
    # --- 4. NEW: Re-index based on 'trialNum' ---
    # Set the DataFrame's index to be (trialNum - 1).
    # This is the new 0-based index you requested.
    sorted_data.index = sorted_data['trialNum'] - 1
    
    # --- 5. Filter for 'correct' trials ---
    # From the re-indexed data, select only rows where 'correct' is 1
    correct_trials = sorted_data[sorted_data['correct'] == 1]
    
    # --- 6. Return the new, 0-based index ---
    # .index returns the new [0, 1, 3, ...] style index
    return correct_trials.index