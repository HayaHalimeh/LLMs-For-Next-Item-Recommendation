
import os
import sys
import random
import re
import json
import difflib
from pprint import pprint
import numpy as np
import logging
import argparse
import torch
from datasets import load_dataset
from collections import defaultdict


from llm_for_nir.src.utils import freeze_seed, check_path
from metrics import eval, generate_rank_tensor

import time


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(),  
    ]
)

logging.info("Script is starting...") 

def main(args):
        
    dir = args.dir
    dataset = args.dataset 
    folder = args.folder
    case=args.case

    ks = args.ks 
    seed = args.seed

    
    freeze_seed(seed)
    logging.info('Configs: %s', args)

    os.environ["PATH"] = dir
    logging.info(f"Current directory set to: {dir}")


    #filter case
    def filter_function(example):
        return example['case'] == case


    #parsing 
    def parse_candidates(data_point):
        parsed_candidates = re.findall(r'"(.*?)(?<!\\)"', data_point)
        parsed_candidates = [item.replace('\\"', '"').strip() for item in parsed_candidates]
        return parsed_candidates
            

    def apply_parse_candidates(example):
        example['parsed_candidates'] = parse_candidates(example['candidates'])
        return example
    

    #permute
    def permute_sample(candidate_list, seed=None):
        if seed is not None:
            random.seed(seed)
        
        sample = random.sample(candidate_list, len(candidate_list))
        return sample
    
    def apply_permute_sample(example, seed):
            example['permuted_candidates'] = permute_sample(example['parsed_candidates'], seed=seed)
            return example
    

    #gold rank
    def calculate_gold_ranks(example):
        permuted_candidates = example['permuted_candidates']
        data_point = example['gold'].strip('"')
        if data_point in permuted_candidates:
            gold_ranks = torch.tensor(permuted_candidates.index(data_point))
        else:
            
            best_match = difflib.get_close_matches(data_point, permuted_candidates, n=1)
            gold_ranks = torch.tensor(permuted_candidates.index(best_match[0]))
        example['gold_rank'] = [gold_ranks]
        return example


                                
    #ranking
    def generate_ranking(example, seed=None):
        candidates = example.get('permuted_candidates', [])
        
        # If no candidates are available, return None
        if not candidates:
            return None  
        
        # Set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Ensure k does not exceed the number of available candidates
        k = len(candidates)
        
        # Randomly select k unique items
        selected_items = random.sample(candidates, k)
        return selected_items

    

    def apply_generate_ranking(example, seed=None):
        example['ranking'] = generate_ranking(example, seed=seed)
        return example



    #create pred rank
    def generate_rank_tensor_hf_ds(example):
        example['pred_rank'] = generate_rank_tensor(example['ranking'], example['permuted_candidates']).tolist()
        return example 
    
    


    #evaluation 
    def eval_hf_ds(example):
        metrics = {}
        pred_rank = torch.tensor(example['pred_rank'])  
        gold_rank = torch.tensor(example['gold_rank']) 
        
        metrics.update(eval(pred_rank, gold_rank, ks))

        example['metrics'] = metrics
        return example



    # save metrics
    def extract_metrics(example, ks):
        metrics = example['metrics']  
        return {f'HR@{k}': metrics.get(f'HR@{k}', 0.0) for k in ks} | {f'MRR@{k}': metrics.get(f'MRR@{k}', 0.0) for k in ks} | {f'NDCG@{k}': metrics.get(f'NDCG@{k}', 0.0) for k in ks}


    
    test_data_path = os.path.join(os.environ.get("PATH"), os.path.join('datasets', dataset, folder))
    test_file = os.path.join(test_data_path , 'test.json')
            
    if not os.path.exists(test_file):
        print("Test_data_path does not exist")
        sys.exit(1)
        
    result_path = os.path.join(os.environ.get("PATH"), os.path.join('results', dataset)) 
    result_dir = os.path.join(result_path , 'random', folder)
    check_path(result_dir)

    result_file = os.path.join(result_dir , 'random_' + case + '_results' + '_seed_' + str(seed) + '.json') 

    logging.info("Test File %s", test_file)
    logging.info("Result File %s", result_file)

    if test_file.endswith(".json"):
        test_data = load_dataset("json", data_files= test_file)
    
        #filter cold/warm
        test_data['train'] = test_data['train'].filter(filter_function)
        
        #shuffle
        test_data["train"] = test_data["train"].shuffle(seed=seed)

        #parse candidates
        test_data = test_data.map(apply_parse_candidates)
        

        #permute
        test_data = test_data.map(lambda example: apply_permute_sample(example, seed=seed))
      
        #gold rank
        test_data = test_data.map(calculate_gold_ranks)
        

        #apply ranking
        start_time = time.time()
        test_data = test_data.map(lambda x: apply_generate_ranking(x, seed= seed))

        end_time = time.time()
        logging.info("-"*30)
        logging.info("Time taken to generate predictions:  %.2f seconds", end_time - start_time)
        logging.info("-"*30)

        #pred rank
        test_data = test_data.map(generate_rank_tensor_hf_ds)
        test_data = test_data['train']        
        
            
        #eval
        test_data = test_data.map(eval_hf_ds)
        #extract metrics
        metrics_list = [extract_metrics(x, ks) for x in test_data]

        # Initialize dictionaries to store metrics
        all_metrics = defaultdict(list)
      
        # Aggregate the metrics
        for metrics in metrics_list:
            for k in ks:
                all_metrics[f'HR@{k}'].append(metrics[f'HR@{k}'])
                all_metrics[f'MRR@{k}'].append(metrics[f'MRR@{k}'])
                all_metrics[f'NDCG@{k}'].append(metrics[f'NDCG@{k}'])



        # Calculate the average for each metric and the invalid output count
        results = {
            'case': case,
            **{f'{metric}@{k}': np.mean(all_metrics[f'{metric}@{k}']) for k in ks for metric in ['HR', 'MRR', 'NDCG']}
        }

        logging.info("Results %s", results)

        with open(result_file, 'w') as f:
            logging.info("Save Metrics", )
            json.dump(results, f, indent=4)
            
        logging.info("Result saved to %s", result_file)
        logging.info("-"*30)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to .")
    parser.add_argument('--dir', type=str, help="The directory to the files.", default= r"./" )
    parser.add_argument('--dataset', type=str, help="The name of the dataset", choices=["ml-1m"], default="ml-1m")  
    parser.add_argument('--folder', type=str, help="The folder of the dataset", default="k_all_c_all")  
    parser.add_argument('--case', type=str, help="Whether cold or warm", choices=["cold", "warm"], default="cold") 
    parser.add_argument('--ks', type=int, help="A list of k for evaluation", default=[1]) 
    parser.add_argument('--seed', type=int, help="The seed for the run", default= 42) 

    args = parser.parse_args()
    main(args)


