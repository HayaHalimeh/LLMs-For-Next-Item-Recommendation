import random
import numpy 
import pandas as pd
import os
import torch
import difflib
import re
import torch

# Function to set the random seed for reproducibility
def freeze_seed(seed: int):
    numpy.random.seed(seed)
    random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    try:
        os.environ["PYTHONASHSEED"] = str(seed)
    except ImportError:
        pass


# Function to read a file and return its lines
def read_file(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines

# Function to check if a path exists and create it if not
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Path {path} sucessfully created!")
           
# Function to save a list of lines to a file
def save_file(file_path: str, content: list):
    with open(file_path, 'w') as f:
        for line in content:
            f.write(line)
            f.write('\n')


# Function to convert lines to a DataFrame
def lines_to_df(lines: list):
    col_str = lines[0]
    columns = [col.strip() for col in col_str.split('\t') if col.strip()]

    data_split = []
    for line in lines[1:]:
        row = [elem.strip() for elem in line.split('\t')]
        if len(row) == len(columns):
            data_split.append(row)
        else:
            print(f"Skipping malformed line: {line.strip()}")

    return pd.DataFrame(data_split, columns=columns)


# Function to sample candidate items
def sample_candidate_items(history, item_list, n=9, seed=None):
    if seed is not None:
        numpy.random.seed(seed)  
    
    history_set = set(history)  
    available_items = [item for item in item_list if item not in history_set]

    if len(available_items) < n:
        return 0  # Return 0 if there aren't enough items to sample

    sampled_items = numpy.random.choice(available_items, n, replace=False).tolist()
    return sampled_items


# Function to randomly select one row per group
def random_select(group, seed=42):
    return group.sample(n=1, random_state=seed)


# Function to format a list of items into a string
def apply_format(item_list):
    return ', '.join(item_list)
     
    

def update_history(df, item_list, seed=42):

    random.seed(seed)
    
    # Function to process history for each row
    def process_history(history):

        updated_history = [item for item in history if item not in item_list]
        if updated_history:
            selected_item = random.choice(updated_history)
            updated_history.remove(selected_item)
            
        return updated_history, selected_item

    # Apply the function to each row's history and split the result into new columns
    df[['history', 'gold']] = df['all_history'].apply(lambda x: pd.Series(process_history(x)))

    return df





