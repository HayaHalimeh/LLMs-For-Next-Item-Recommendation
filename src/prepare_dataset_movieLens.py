import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import argparse
from sklearn.model_selection import train_test_split
from llm_for_nir.src.utils import freeze_seed, read_file, lines_to_df, sample_candidate_items, apply_format, check_path
from retrieval_methods.bm25 import BM25ItemRetriever
from retrieval_methods.pop import PopularityRetriever
from retrieval_methods.user_filtering import UserBasedFilteringRetriever

import time

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)  
pd.set_option('display.expand_frame_repr', False)  
pd.options.mode.copy_on_write = True 


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(),
        ])


logging.info("Script is starting...")


def report_general_statistics(data):
    logging.info("Reporting general statistics...")
    logging.info("Number of unique users: %d", len(data['user_id'].unique()))
    logging.info("Number of unique items: %d", len(data['item_id'].unique()))
    logging.info("Number of interactions: %d", len(data))
    logging.info("Average interactions per user: %f", len(data) / len(data['user_id'].unique()))
    logging.info("Average interactions per item: %f", len(data) / len(data['item_id'].unique()))
    logging.info("-"*20)


def report_statistics(data,  case):
    logging.info("Reporting statistics...")
    logging.info("Number of interactions: %d", len(data))
    logging.info("Number of unique users: %d", len(data['user_id'].unique()))
    
    logging.info("Number of total warm cases: %d", len(data[data[case] == 'warm']))
    logging.info("Number of total cold cases: %d", len(data[data[case] == 'cold']))
    logging.info("Ratio warm to all: %f", len(data[data[case] == 'warm'])/len(data))
    logging.info("Ratio cold to all: %f", len(data[data[case] == 'cold'])/len(data))

    logging.info("Number of unique items for warm items: %d", len(data[data[case] == 'warm']['gold'].unique()))
    logging.info("Number of unique items for cold items: %d", len(data[data[case] == 'cold']['gold'].unique()))
    
    logging.info("Number of unique users for warm items: %d", len(data[data[case] == 'warm']['user_id'].unique()))
    logging.info("Number of unique users for cold items: %d", len(data[data[case] == 'cold']['user_id'].unique()))
    logging.info("-"*20)

                 

# Function to get movies for a user
def get_user_movies(df, items_dict):
    return items_dict.get(df, "")



def main(args):
    seed = args.seed
    dir = args.dir
    dataset = args.dataset 

    fraction = args.fraction
    min_interactions = args.min_interactions
    kind = args.kind
    candidate_type= args.candidate_type
    icl = args.icl
    number_of_candidates = args.number_of_candidates
    top_n_users = args.top_n_users

    candidates_seed = args.candidates_seed

    
    logging.info('Configs: %s', args)
    logging.info('Case: %s',  'k_' + kind + '_c_' + candidate_type ) 

    freeze_seed(seed)
    os.environ["PATH"] = dir 

    path_env = os.path.join(os.environ.get("PATH"), 'datasets')


    if not os.path.exists(path_env):
        print("Directory does not exist")
        sys.exit(1)
    
   
    interactions = read_file(os.path.join(path_env, dataset, 'atomic', dataset + '.inter'))
    interactions = lines_to_df(interactions)
    interactions.columns = [col.split(':')[0] for col in interactions.columns]

    
    items = read_file(os.path.join(path_env, dataset, 'atomic', dataset + '.item'))
    items= lines_to_df(items)
    items.columns = [col.split(':')[0] for col in items.columns]

    logging.info("Read Data and transformed into df")

    # Sort the data by user id and timestamp
    interactions = interactions.sort_values(by=['user_id', 'timestamp'])
    data = pd.merge(interactions, items, on=['item_id'], how='left')
    data = data[['user_id', 'item_id', 'timestamp', 'movie_title']]

    logging.info("Remove duplicates")
    # Some titles occur with different item ids, keep the most frequent item_id 
    # Drop duplicates
    most_frequent = data.groupby('movie_title')['item_id'].agg(lambda x: x.mode()[0])
    # Filter the DataFrame to only keep rows where 'item_id' matches the most frequent item_id for each movie_title
    data = data[data.apply(lambda row: row['item_id'] == most_frequent[row['movie_title']], axis=1)]
    data['cumulative_interactions'] = data.groupby('user_id').cumcount() + 1
    report_general_statistics(data)


    if dataset.startswith('ml'): 
        title_col = 'movie_title'
        data[title_col] = data[title_col].apply(lambda x: f'"{x}"')

        #create dataframe with user id and all history for each user
        all_history_df = data.groupby('user_id')[title_col].apply(list).reset_index().rename(columns={title_col: 'all_history'})


    if kind == "pos_only":
        # Remove disliked items <3
        data = data[data['rating'].isin(['3', '4', '5'])]
        logging.info("Stats after removing disklikes")
        report_general_statistics(data)

    
    # Sample fraq% of all items after ['cumulative_interactions'] > 5 (keep first 5 items for each user)
    temp_after5 = data[data['cumulative_interactions'] > 5]
    temp_before5 = data[data['cumulative_interactions'] <= 5]
    
    items_for_sampling_cold = [i for i in temp_after5['item_id'].unique() if i not in temp_before5['item_id'].unique()]
    
    total_items = len(data['item_id'].unique())
    dynamic_fraction = fraction * total_items / len(items_for_sampling_cold)
    dynamic_fraction = min(dynamic_fraction, 1.0)
    logging.info('dynamic_fraction: %f', dynamic_fraction)
    
    cold_sampled_items = pd.unique(pd.Series(items_for_sampling_cold).sample(frac=dynamic_fraction, random_state=seed).values).tolist()
    # Assert sampled_items are sampled from later interactions
    intersected_items = np.intersect1d(temp_before5['item_id'].unique().tolist(), cold_sampled_items)
    assert len(intersected_items) == 0, "temp_before5 contains cold items"

    logging.info('Number of cold items sampled: %d', len(cold_sampled_items))
    logging.info('Percentage sampled item/ all items: %f', len(cold_sampled_items) / len(data['item_id'].unique()))

    
    # Assert that the rounded fraction matches the expected fraction (.1% of all items), with a detailed error message if it fails
    fraction_calculated = round(len(cold_sampled_items) / 
                                len(data['item_id'].unique()), 2)
   
    assert np.isclose(fraction_calculated, fraction, atol=1e-2), f"Expected {fraction}, but got {fraction_calculated}"



    # Determine all appearance of each sampled item for each user
    temp = data[data['item_id'].isin(cold_sampled_items)]
    cold_items_appearances = temp.groupby(['user_id', 'item_id']).head(1).reset_index(drop=True).sort_values(by=['user_id', 'timestamp'])
    
    # keep only the very first appearance of a cold item for each user
    cold_items_first_appearances = cold_items_appearances.groupby('user_id').head(1).reset_index(drop=True)
    

    # Assert cold_items_first_appearances contains only items from sampled_items 
    diff_items = [item for item in cold_items_first_appearances['item_id'].unique().tolist() if item not in cold_sampled_items]
    assert len(diff_items) == 0, "cold_items_first_appearances contains warm items and not items only from cold_sampled_items"

    
    # Merge with warm cases 
    merged = pd.merge(data, cold_items_first_appearances[['user_id', 'cumulative_interactions']], on='user_id', how='left', suffixes=('', '_cutoff'))
    merged.fillna({'cumulative_interactions_cutoff': int(len(data))}, inplace=True)
    merged['case'] = np.where(
        merged['cumulative_interactions_cutoff'] == int(len(data)), 
        'warm', 
        'cold'
    )
    assert len(data) == len(merged), 'Probelm with merging'
    

    logging.info("Filter up to cold or last intercation")
    # Keep data only up to cold item to avoid data leakage, for warm cases retain all. 
    filtered_data_up_to_cold = merged[merged['cumulative_interactions'] <= merged['cumulative_interactions_cutoff']]
    

    logging.info("Min interaction filtering")
    # Keep only users with at least min_interactions
    filtered_data = filtered_data_up_to_cold[filtered_data_up_to_cold['user_id'].map(filtered_data_up_to_cold['user_id'].value_counts()) > min_interactions].reset_index(drop=True)    
    report_general_statistics(filtered_data)


    interactions_per_user = filtered_data['user_id'].value_counts().values.tolist()
    logging.info('Percentage of users with history length > 50 to all users: %f',len([i for i in interactions_per_user if i > 50])/len(interactions_per_user))

    logging.info("Data stats cold & warm")
    logging.info("Nr of cold cases %s", len(filtered_data[filtered_data['case'] == 'cold']))
    logging.info("Nr of warm cases %s", len(filtered_data[filtered_data['case'] == 'warm']))


    # Keep the last interaction for each user to use as the ground truth for test & val set
    last_interactions = filtered_data.groupby('user_id').tail(1).reset_index(drop=False)
    
    # Keep the rest of the data as the training set
    data_wo_li = filtered_data.drop(last_interactions['index']).reset_index(drop=True)
   

    # Assert to ensure no cold items in the training set
    intersected_items = np.intersect1d( data_wo_li['item_id'].unique().tolist(), cold_sampled_items)
    assert len(intersected_items) == 0, "Train data contains cold items"

    cols_to_keep = ['user_id', 'item_id', 'timestamp', 'movie_title', 'case']
    last_interactions = last_interactions[cols_to_keep] 
    data_wo_li = data_wo_li[cols_to_keep] 

    last_interactions = last_interactions.rename(columns={title_col:'gold'})
   
    logging.info("Group Data")
    #Reshape data_wo_li so that each user has one row, its user id and its history  
    data_grouped = data_wo_li.groupby('user_id')[title_col].apply(list).reset_index().rename(columns={title_col: 'history'})
    data_grouped = pd.merge(data_grouped, last_interactions, on='user_id', suffixes=('', ''))

   
    logging.info("Create candidate sets")
    # Add all_history_df columns to create candidate sets with only one ground truth
    data_grouped = pd.merge(data_grouped, all_history_df, on='user_id', how='left')
    
    if icl:
        number_of_candidates = number_of_candidates * 2
        top_n_users = top_n_users 
    else:
        number_of_candidates = number_of_candidates
        top_n_users = top_n_users


    start_time = time.time()
    #For cold cases: the cold gt item in test shouldnt appear in train or in train candidates to prevent leakage.
    item_titles = pd.unique(pd.Series(data[title_col])).tolist()
    cold_gt = data_grouped[data_grouped['case']== 'cold']['gold'].unique().tolist()  
    items_wo_cold_gt = [i for i in item_titles if i not in cold_gt]
    
    logging.info('Number of cold ground truth: %d', len(cold_gt))
    logging.info('Number of items_wo_cold_gt: %d', len(items_wo_cold_gt))
    assert len(item_titles) - len(items_wo_cold_gt) == len(cold_gt), "Candidate set creation in all doesnt add up!"

   
    if candidate_type == 'all':
       data_grouped['candidates'] = data_grouped['all_history'].apply(lambda x: sample_candidate_items(x, items_wo_cold_gt, n=number_of_candidates, seed=candidates_seed))


    logging.info("Add gold to the candidate sets")
    def add_gold_to_candidates(row):
        return list(set(row['candidates'] + [row['gold']]))
    

    # Add gold to the candidate sets
    data_grouped['candidates'] = data_grouped.apply(add_gold_to_candidates, axis=1)
    
    end_time = time.time()
    logging.info("-"*30)
    logging.info("Time taken to create candidate sets: %.2f seconds", end_time - start_time)
    logging.info("-"*30)
    
    # Remove all history column after creating the candidate sets
    data_grouped = data_grouped[['user_id', 'history', 'candidates', 'gold', 'case' ]]

    # Create Test & Val 80:10:10
    logging.info("Create Train & Val & Test ratio 80:10:10")

    def create_splits(df, split_ratio=(0.8,0.1,0.1)):
        '''
        Function to create train, valid, and test splits for both 'cold' and 'warm' cases
        based on unique 'gold' values.
        '''

        def split_case(case_df):
            # Get unique 'gold' values
            unique_gold = case_df['gold'].unique()
            
            # First, split into train and remaining (validation + test)
            gold_train, gold_remaining = train_test_split(unique_gold, train_size=split_ratio[0], random_state=seed)
            
            # Then split the remaining gold values into validation and test
            gold_test, gold_valid = train_test_split(gold_remaining, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)
        
           
            # Create the final splits by filtering the original dataframe based on 'gold' values
            train_df = case_df[case_df['gold'].isin(gold_train)]
            valid_df = case_df[case_df['gold'].isin(gold_valid)]
            test_df = case_df[case_df['gold'].isin(gold_test)]
            
            return train_df, valid_df, test_df

        # Separate 'cold' and 'warm' cases
        train_cold, valid_cold, test_cold = split_case(df[df['case'] == 'cold'])
        train_warm, valid_warm, test_warm = split_case(df[df['case'] == 'warm'])
        
        # Combine the splits for 'cold' and 'warm'
        train_df = pd.concat([train_cold, train_warm])
        valid_df = pd.concat([valid_cold, valid_warm])
        test_df = pd.concat([test_cold, test_warm])
        
        return train_df, valid_df, test_df

  
    train, valid, test = create_splits(data_grouped)


    logging.info("-"*20)
    logging.info("Assertions")

    intersected_items = np.intersect1d(train['user_id'].unique(), valid['user_id'].unique())
    assert len(intersected_items) == 0, "Train data contains valid users"

    intersected_items = np.intersect1d(train['user_id'].unique(), test['user_id'].unique())
    assert len(intersected_items) == 0, "Train data containstest test users"

    intersected_items = np.intersect1d(valid['user_id'].unique(), test['user_id'].unique())
    assert len(intersected_items) == 0, "Overlap between valid & test users"

    train_history = train.explode('history')[['user_id', 'history']].rename(columns={'history': 'item_id'})
    train_history = train_history['item_id'].unique().tolist()

    train_candidates = train.explode('candidates')[['user_id', 'candidates']].rename(columns={'candidates': 'item_id'})
    train_candidates = train_candidates['item_id'].unique().tolist()



    valid_history = valid.explode('history')[['user_id', 'history']].rename(columns={'history': 'item_id'})
    valid_history = valid_history['item_id'].unique().tolist()

    valid_candidates = valid.explode('candidates')[['user_id', 'candidates']].rename(columns={'candidates': 'item_id'})
    valid_candidates = valid_candidates['item_id'].unique().tolist()


    test_history = test.explode('history')[['user_id', 'history']].rename(columns={'history': 'item_id'})
    test_history = test_history['item_id'].unique().tolist()

    test_candidates = test.explode('candidates')[['user_id', 'candidates']].rename(columns={'candidates': 'item_id'})
    test_candidates = test_candidates['item_id'].unique().tolist()


    train_cold = train[train['case']== 'cold']
    valid_cold = valid[valid['case']== 'cold']
    test_cold = test[test['case']== 'cold']

    train_golds = train_cold['gold'].unique().tolist()
    valid_golds = valid_cold['gold'].unique().tolist()
    test_golds = test_cold['gold'].unique().tolist()
    

    logging.info("Train histories do not contain any test or valid cold gorund truth.")
    intersected_items = np.intersect1d(train_history, valid_golds)
    assert len(intersected_items) == 0, "Train histories contains valid cold ground truth"

    intersected_items = np.intersect1d(train_history, test_golds)
    assert len(intersected_items) == 0, "Train histories contains test cold ground truth"

    logging.info("Valid histories do not contain any test cold gorund truth.")
    intersected_items = np.intersect1d(valid_history, test_golds)
    assert len(intersected_items) == 0, "Train histories contains test cold ground truth"

    logging.info("Test histories do not contain any test cold gorund truth.")
    intersected_items = np.intersect1d(test_history, test_golds)
    assert len(intersected_items) == 0, "Test histories contains test cold ground truth"


    logging.info("Train cold ground truths  do not contain any valid or test cold gorund truth.")
    intersected_items = np.intersect1d(train_golds, valid_golds)
    assert len(intersected_items) == 0, "Train cold ground truths contain valid cold ground truth"

    intersected_items = np.intersect1d(train_golds, test_golds)
    assert len(intersected_items) == 0, "Train cold ground truths contain test cold ground truth"

    logging.info("Valid cold ground truths do not contain any  test cold gorund truth.")
    intersected_items = np.intersect1d(valid_golds, test_golds)
    assert len(intersected_items) == 0, "Valid cold ground truths contain test cold ground truth"

    logging.info("Train candidates do not contain any valid or test cold gorund truth.")
    intersected_items = np.intersect1d(train_candidates, test_golds)
    assert len(intersected_items) == 0, "Train candidates contain test cold ground truth"

    intersected_items = np.intersect1d(train_candidates, valid_golds)
    assert len(intersected_items) == 0, "Train candidates contain valid cold ground truth"

    logging.info("Valid candidates do not contain any test cold gorund truth.")
    intersected_items = np.intersect1d(valid_candidates, test_golds)
    assert len(intersected_items) == 0, "Valid candidates contain test cold ground truth"


    logging.info("-"*20)


    logging.info("Train")
    report_statistics(train, case='case')

    logging.info("Valid")
    report_statistics(valid, case='case')

    logging.info("Test")
    report_statistics(test, case='case')

    
    logging.info("Format the output from list to str")
    # Format the output from list to str for JSON
    for df in [train, valid, test]:
        df['history'] = df['history'].apply(apply_format)
        df['candidates'] = df['candidates'].apply(apply_format)

    
 
    logging.info("-"*20)
    to_path = os.path.join(os.environ.get("PATH"), 'datasets', dataset, 'k_' + kind + '_c_' + candidate_type ) 
    
    if icl:
        number_of_candidates = int(number_of_candidates/2)
        

    to_path = os.path.join(to_path, str(number_of_candidates + 1), str(candidates_seed))

    if icl: 
        to_path =  os.path.join(to_path, 'icl')

    check_path(to_path)


    with open(os.path.join(to_path,'test.json'), 'w') as f:
        json.dump(test.to_dict(orient='records'), f, indent=4)

    with open(os.path.join(to_path,'valid.json'), 'w') as f:
        json.dump(valid.to_dict(orient='records'), f, indent=4)

    with open(os.path.join(to_path,'train.json'), 'w') as f:
            json.dump(train.to_dict(orient='records'), f, indent=4)

    logging.info("-"*30)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to prepare the datasets.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed value')
    parser.add_argument('--dir', type=str, help="The directory to the files.", default="./")
    parser.add_argument('--dataset', type=str, help="The dataset name.", default="ml-1m")
    parser.add_argument('--fraction', type=float, help="The fraction of the cold start items for creation", default=.1) 
    parser.add_argument('--min_interactions', type=int, help="The minimum number of interactions for a user.", default=5)
    parser.add_argument('--kind', type=str, help="Whether all history or only liked items", choices=['all', 'pos_only'] , default= 'all') 
    parser.add_argument('--candidate_type', type=str, help="The type of the candidate list", choices=['all'] , default= 'all')
    parser.add_argument('--top_n_users', type=int, help="Number of users for user filtering and bm25" , default= 10)
    parser.add_argument('--number_of_candidates', type=int, help="Number of the candidate items without ground truth item" , default= 9)
    parser.add_argument('--icl', type=bool, help="if the candidate set is created for in-context-learning" , default= False)
    parser.add_argument('--candidates_seed', type=int, help="if seed for random candidates" , default= 42)

    args = parser.parse_args()
    main(args)


