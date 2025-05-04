
import torch as tc
from transformers import BertTokenizer, BertForNextSentencePrediction
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
from rapidfuzz import process, fuzz
from collections import defaultdict

from llm_for_nir.src.utils import freeze_seed, check_path
from metrics import eval, generate_rank_tensor

import time
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(), 
    ]
)

logging.info("Script is starting...") 

# Simplified PLM class 
class PLM:
    def __init__(self, model_name, device='cuda'):
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
    
    def __call__(self, input_ids, masks=None, segs=None):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=masks, 
            token_type_ids=segs
        )
        return outputs.logits


class PairwiseNSP(tc.nn.Module):
    def __init__(self, meta, plm, device, *args, **kwargs):
        super(PairwiseNSP, self).__init__()
        self.plm = PLM(plm, device=device) if isinstance(plm, str) else plm
        self.sepidx = self.plm.tokenizer.sep_token_id
        self.clsidx = self.plm.tokenizer.cls_token_id
        self.device = device
        self.to(device)
    
    def forward(self, pair_ids, pair_masks, pair_segs, **args):
        logits = self.plm(pair_ids, masks=pair_masks, segs=pair_segs)
        probabilities = tc.softmax(logits, dim=-1)
        is_next_probs = probabilities[:, 0]  
        return is_next_probs.flatten()



def main(args):
        
    dir = args.dir
    dataset = args.dataset 
    folder = args.folder
    case=args.case
    model_id = args.model_id
    ks = args.ks
    seed = args.seed
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name



    wandb.init(
    project= wandb_project,   
    name= wandb_run_name)


    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        
    # Instantiate the PairwiseNSP model
    meta = None  
    plm_model_name = model_id
    model = PairwiseNSP(meta, plm_model_name, device)
    tokenizer = BertTokenizer.from_pretrained(plm_model_name)


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
    def generate_ranking(example, max_length= 512):

        model.eval()
        candidates = example.get('permuted_candidates', [])
        max_k = len(candidates)

        if not candidates:
            return None  # Return None if no candidates available
        history = example.get('history', '')
        if not history:
            return None
        history = history[-512:]
                
        # Store scores for each candidate
        candidate_scores = {}

        # Iterate over each candidate item
        for item in candidates:            
            # Encode the pair of sentences
            encoding = tokenizer.encode_plus(
                history,
                item,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            pair_ids = encoding['input_ids'].to(device)
            pair_masks = encoding['attention_mask'].to(device)
            pair_segs = encoding['token_type_ids'].to(device)
            
            # Compute the probability
            with tc.no_grad():
                is_next_probs = model(pair_ids, pair_masks, pair_segs)
            
            probability = is_next_probs.item()
            candidate_scores[item] = probability
            

        # Sort candidates by their scores in descending order and retrieve top-k
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_candidates = [candidate for candidate, score in sorted_candidates[:max_k]]
        return top_k_candidates

        

    def apply_generate_ranking(example):
        example['ranking'] = generate_ranking(example)
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
    result_dir = os.path.join(result_path , 'nsp', folder)
    check_path(result_dir)

    result_file = os.path.join(result_dir , 'nsp_' + case + '_results' + '_seed_' + str(seed) + '.json') 

    logging.info("Test File %s", test_file)
    logging.info("Result File %s", result_file)

    if test_file.endswith(".json"):
        test_data = load_dataset("json", data_files= test_file)
    
    if case != None:
        test_data['train'] = test_data['train'].filter(filter_function)
    
    
    #shuffle
    test_data["train"] = test_data["train"].shuffle(seed=seed)
 
    logging.info("Test set size")
    logging.info(len(test_data["train"] ))


    #parse candidates
    test_data = test_data.map(apply_parse_candidates)
    
    #premute
    test_data = test_data.map(lambda example: apply_permute_sample(example, seed=seed))
    
    #gold rank
    test_data = test_data.map(calculate_gold_ranks)
    
    
    #apply ranking
    start_time = time.time()
    test_data = test_data.map(apply_generate_ranking)

    end_time = time.time()
    inference_time = time.time() - start_time
    wandb.log({"Inference Latency (s)": inference_time})
    logging.info("-"*30)

    logging.info("-"*30)
    logging.info("Time taken to generate predictions: %.2f seconds", end_time - start_time)
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
    logging.info("-"*20)

    wandb.finish()


   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to .")
    parser.add_argument('--dir', type=str, help="The directory to the files.", default= r"./" )
    parser.add_argument('--dataset', type=str, help="The name of the dataset", choices=['ml-1m'], default='ml-1m')  
    parser.add_argument('--folder', type=str, help="The folder of the dataset", default="k_all_c_all")  
    parser.add_argument('--case', type=str, help="Whether cold or warm", choices=['cold', 'warm'], default='cold') 
    parser.add_argument('--ks', type=int, help="A list of k for evaluation", default=[1]) 
    parser.add_argument('--seed', type=int, help="The seed for the run", default= 42) 
    parser.add_argument('--model_id', type=str, help="The model name from Hf", default= "bert-base-cased") 
    parser.add_argument('--wandb_project', type=str, help="Weights and Biases project name.", default="")
    parser.add_argument('--wandb_run_name', type=str, help="Weights and Biases run name.", default="")
    args = parser.parse_args()
    main(args)


