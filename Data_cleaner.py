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

def get_incorrect_trials(data, session_name='session_00'):
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
    incorrect_trials = sorted_data[sorted_data['correct'] == 0]
    
    # --- 6. Return the new, 0-based index ---
    # .index returns the new [0, 1, 3, ...] style index
    return incorrect_trials.index

def get_correct_df(units,trials):
    num_sessions = 27
    session_name_list = [f"session_{i:02d}" for i in range(num_sessions)]
    units_new = units.head(0)
    trials_new = trials.head(0)
    num_of_correct = []
    num_of_total_trials = []
    for i in session_name_list:
        units_cur_session = units[units['session'] == i]
        trials_cur_session = trials[trials['session'] == i]
        correct_index = get_correct_trials(data = trials, session_name=i)
        num_of_correct.append(len(correct_index))
        num_of_total_trials.append(trials_cur_session.shape[0])
        
        trials_cur_session = trials_cur_session.iloc[correct_index]
        trials_new = pd.concat([trials_new,trials_cur_session],ignore_index=True)

        count = 0
        for ind in units_cur_session.index:
            original_matrix = units_cur_session.at[ind,'spkMtx']
            new_filtered_matrix = original_matrix[correct_index]
            units_cur_session.at[ind,'spkMtx'] = new_filtered_matrix
            units_new = pd.concat([units_new,units_cur_session],ignore_index=True)
            count += 1

    return units_new,trials_new,num_of_correct,num_of_total_trials

def get_incorrect_df(units,trials):
    num_sessions = 27
    session_name_list = [f"session_{i:02d}" for i in range(num_sessions)]
    units_new = units.head(0)
    trials_new = trials.head(0)
    num_of_incorrect = []
    num_of_total_trials = []
    for i in session_name_list:
        units_cur_session = units[units['session'] == i]
        trials_cur_session = trials[trials['session'] == i]
        incorrect_index = get_incorrect_trials(data = trials, session_name=i)
        num_of_incorrect.append(len(incorrect_index))
        num_of_total_trials.append(trials_cur_session.shape[0])
        
        trials_cur_session = trials_cur_session.iloc[incorrect_index]
        trials_new = pd.concat([trials_new,trials_cur_session],ignore_index=True)

        count = 0
        for ind in units_cur_session.index:
            original_matrix = units_cur_session.at[ind,'spkMtx']
            new_filtered_matrix = original_matrix[incorrect_index]
            units_cur_session.at[ind,'spkMtx'] = new_filtered_matrix
            units_new = pd.concat([units_new,units_cur_session],ignore_index=True)
            count += 1

    return units_new,trials_new,num_of_incorrect,num_of_total_trials