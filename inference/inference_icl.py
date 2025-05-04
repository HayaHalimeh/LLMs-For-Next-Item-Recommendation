
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
from rapidfuzz import process, fuzz
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, AutoModelForCausalLM


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


def main(args):
        
    dir = args.dir
    dataset = args.dataset 
    folder = args.folder
    case=args.case
    prompts_path = args.prompts
    prompt_type = args.prompt_type

    model_id = args.model_id

    ks = args.ks 
    seed = args.seed
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name


    wandb.init(
    project= wandb_project,   
    name= wandb_run_name)

    set_seed(seed)
    
    logging.info('Configs: %s', args)

    os.environ["PATH"] = dir
    logging.info(f"Current directory set to: {dir}")



    #------ Functions
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
    
    def split_history(history):
        items = history.strip("\"").split("\", \"")
        # Last item
        last_item = items[-1]
        # All items before the last one
        previous_items = items[:-1]
        return previous_items, last_item
    

    def add_candidates_new(example, original_data):
        # Find the matching row in the original data based on user_id
        matching_row = next((row for row in original_data if row["user_id"] == example["user_id"]), None)
        if matching_row:
            example["candidates_new"] = matching_row["parsed_candidates"]
        return example
    
    def separate_candidates_icl(example):
        # Select items from parsed_candidates that are not in candidates_new
        candidates_icl = [candidate for candidate in example["parsed_candidates"] if candidate not in example["candidates_new"]]
        
        # Ensure only 9 items are included
        candidates_icl = candidates_icl[:9]
        
        # Add new column to the example
        example["candidates_icl"] = candidates_icl
        return example
        

    def apply_permute_sample(example, col_title, seed):
        example['permuted_'+ col_title] = permute_sample(example[col_title], seed=seed)
        return example


    def to_string(permute_sample):
        return ', '.join([f'"{candidate}"' for candidate in permute_sample])



    def apply_to_string(example, col_title):
        example[col_title + '_string'] = to_string(example[col_title])
        return example    
    


    #gold rank
    def calculate_gold_ranks(example, col_title):
        permuted_candidates = example[col_title]
        data_point = example['gold'].strip('"')
        if data_point in permuted_candidates:
            gold_ranks = torch.tensor(permuted_candidates.index(data_point))
        else:
            best_match = difflib.get_close_matches(data_point, permuted_candidates, n=1)
            gold_ranks = torch.tensor(permuted_candidates.index(best_match[0]))
        example['gold_rank'] = [gold_ranks]
        return example


    #prompting
    def formatting_prompts_func_bookcrossing(examples, instruction):
        histories = examples["previous_items_string"]
        candidates_icl_set = examples["permuted_candidates_icl_string"]
        candidates_new_set = examples["permuted_candidates_new_string"]
        last_items = examples["last_item"]

        messages = []
        for history, candidate_icl, last_item, candidate_new in zip(histories, candidates_icl_set, last_items, candidates_new_set):
            user_content = instruction.format(history, candidate_icl, last_item, candidate_new)

            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for recommendations."
                },
                {
                    "role": "user", 
                    "content": user_content
                },
            ]
            messages.append(chat)
        return { "messages" : messages }
    
    

    
    def formatting_prompts_func_movielens(examples, instruction):
        
        histories = examples["previous_items_string"]
        candidates_icl_set = examples["permuted_candidates_icl_string"]
        candidates_new_set = examples["permuted_candidates_new_string"]
        last_items = examples["last_item"]

        messages = []
        for history, candidate_icl, last_item, candidate_new in zip(histories, candidates_icl_set, last_items, candidates_new_set):
            user_content = instruction.format(history, candidate_icl, last_item, candidate_new)

            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for recommendations."
                },
                {
                    "role": "user", 
                    "content": user_content
                },
            
            ]
            messages.append(chat)
        return { "messages" : messages }



    #ranking
    def generate_ranking(example):
        with torch.no_grad():
            inputs = tokenizer(
                example['tokenized_messages'],
                return_tensors="pt",
                max_length=4096,
                truncation=True,
                ).to("cuda")

            with torch.cuda.amp.autocast():
                generated_sents = list()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p = .9,
                    temperature= .6
                )
            output = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
            output = output.split('<|end_header_id|>:\n')[-1] 
            return output 
        

    def apply_generate_ranking(example):
        example['ranking'] = generate_ranking(example)
        return example


    #parse and filter LLM response using fuzzy matching
    def parse_response(example):
        output = example['ranking']  
        candidates = example['permuted_candidates_new']  

        # Remove the format and extract the relevant lines
        lines = output.split("assistant")[-1].strip()  
        lines = lines.strip().split('\n')
        
        # Extract items from the lines
        items = [re.sub(r'\d+\.\s*', '', line).strip().replace('<|eot_id|>', '') for line in lines]

        # Filter items using fuzzy matching
        filtered_items = []
        for item in items:
            match = process.extractOne(item.replace('"', ''), candidates, scorer=fuzz.token_set_ratio, score_cutoff=90)
            if match:
                matched_item = match[0]
                if matched_item not in filtered_items:
                    filtered_items.append(matched_item)
        
        
        # Store the filtered items in the example
        example['parsed_ranking'] = filtered_items
        return example
    

    #extract pred rank
    def generate_rank_tensor_hf_ds(example):
        example['pred_rank'] = generate_rank_tensor(example['parsed_ranking'], example['permuted_candidates_new']).tolist()
        return example 
    

    #evaluation 
    def eval_hf_ds(example):
        metrics = {}
        pred_rank = torch.tensor(example['pred_rank'])  
        gold_rank = torch.tensor(example['gold_rank']) 
        metrics.update(eval(rank = pred_rank, labels =gold_rank, ks = ks))
        example['metrics'] = metrics
        return example


    def count_invalid_outputs(example):
        parsed_ranking = example['parsed_ranking']
        permuted_candidate = example['permuted_candidates_new']
        
        invalid_output_metrics = {}

        for k in ks:
            # Consider only the top-k predictions
            top_k_predictions = parsed_ranking[:k] if parsed_ranking else []
          
            if not top_k_predictions:
                # If no items are predicted, assign a penalty equal to k
                invalid_output_count = k
            else:
                # Count hallucinated items (items not in permuted_candidates) in the top-k
                invalid_output_count = sum(1 for item in top_k_predictions if item not in permuted_candidate)
            
            invalid_output_metrics[f'invalid_output@{k}'] = invalid_output_count
        
        # Store the invalid output metrics in the example
        example['invalid_output_metrics'] = invalid_output_metrics
        return example


    # save metrics
    def extract_metrics(example, ks):
        metrics = example['metrics'] 
        invalid_output_metrics=example['invalid_output_metrics'] 
        return {f'HR@{k}': metrics.get(f'HR@{k}', 0.0) for k in ks} | {f'MRR@{k}': metrics.get(f'MRR@{k}', 0.0) for k in ks} | {f'NDCG@{k}': metrics.get(f'NDCG@{k}', 0.0) for k in ks} | {f'invalid_output@{k}': invalid_output_metrics.get(f'invalid_output@{k}', 0.0) for k in ks}

    #Load Prpmpts
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
       

    def add_item_to_candidates(example, col_title, new_item):
        clean_item = example[new_item].strip('"')
        return {col_title: example[col_title] + [clean_item]}


    #model & tokenizer
    PAD_TOKEN= "<|pad|>"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True) # add_eos_token=True,
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    tokenizer.padding_side = "right" 


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    

    test_data_path = os.path.join(os.environ.get("PATH"), os.path.join('datasets', dataset, folder))    
    original_test_path = test_data_path.split("/icl")[0]

    test_file = os.path.join(test_data_path , 'test.json')
    original_test_file = os.path.join(original_test_path , 'test.json')

    if not os.path.exists(test_file):
        print(f"Test file {test_file} does not exist")
        sys.exit(1)

    
    result_path = os.path.join(os.environ.get("PATH"), os.path.join('results', dataset)) 
    result_dir = os.path.join(result_path , 'icl', folder) #.split('/')[0])
    check_path(result_dir)

    result_file = os.path.join(result_dir, prompt_type + '_' + case + '_results' + '_seed_' + str(seed) + '.json') 

    logging.info("Test File %s", test_file)
    logging.info("Original Test File %s", original_test_file)
    logging.info("Result File %s", result_file)


    test_data = load_dataset("json", data_files= test_file)
    original_test_data = load_dataset("json", data_files= original_test_file)

    #filter cold/warm
    if case != None:
        test_data['train'] = test_data['train'].filter(filter_function)
        original_test_data['train'] = original_test_data['train'].filter(filter_function)
        
    
    #parse candidates
    test_data = test_data.map(apply_parse_candidates)
    original_test_data = original_test_data.map(apply_parse_candidates)


    #seperate candidates
    test_data["train"] = test_data["train"].map(
        lambda x: add_candidates_new(x, original_test_data["train"]))


    test_data["train"] = test_data["train"].map(separate_candidates_icl)


    # icl ground truth
    test_data = test_data.map(lambda row: {
        "previous_items": split_history(row["history"])[0],
        "last_item": split_history(row["history"])[1]
    })

    test_data = test_data.map(lambda example: apply_to_string(example, col_title='previous_items'))

    #Add last item/gold to candidate lists
    test_data = test_data.map(lambda example: add_item_to_candidates(example, col_title='candidates_icl', new_item='last_item'))

    
    #permute
    test_data = test_data.map(lambda example: apply_permute_sample(example, 'candidates_icl', seed=seed))
    test_data = test_data.map(lambda example: apply_permute_sample(example, 'candidates_new', seed=seed))
    

    #format permuted candidate for llm (to string)
    test_data = test_data.map(lambda example: apply_to_string(example, col_title='permuted_candidates_icl'))
    test_data = test_data.map(lambda example: apply_to_string(example, col_title='permuted_candidates_new'))


    #gold rank
    test_data = test_data.map(lambda example: calculate_gold_ranks(example, col_title='permuted_candidates_new'))

    #shuffle
    #test_data["train"] = test_data["train"].shuffle(seed=None)


    instruction = prompts[prompt_type]

    #prompt format
    if dataset.startswith('ml'):             
        test_data = test_data["train"].map(
            lambda examples: formatting_prompts_func_movielens(examples, instruction), 
            batched=True)
    else:
        test_data = test_data["train"].map(
            lambda examples: formatting_prompts_func_bookcrossing(examples, instruction),
            batched = True)

    test_data = test_data.map(lambda x: {"tokenized_messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)})

    
    #apply ranking
    start_time = time.time()
    test_data = test_data.map(apply_generate_ranking)
    
    #parse ranking
    test_data = test_data.map(parse_response)

    end_time = time.time()
    inference_time = time.time() - start_time
    wandb.log({"Inference Latency (s)": inference_time})

    logging.info("-"*30)
    logging.info("Time taken to generate predictions: %.2f seconds", end_time - start_time)
    logging.info("-"*30)

    #parse ranking
    test_data = test_data.map(parse_response)

    #pred rank
    test_data = test_data.map(generate_rank_tensor_hf_ds)
   

    logging.info("Start evaluation")
    logging.info(torch.cuda.memory_summary())
    
    #eval
    test_data = test_data.map(eval_hf_ds)


    #invalid output
    test_data = test_data.map(count_invalid_outputs)

    logging.info("test example %s", test_data[-1])
    
    #extract metrics
    metrics_list = [extract_metrics(x, ks) for x in test_data]
    
    
    # Initialize dictionaries to store metrics
    all_metrics = defaultdict(list)
    # Aggregate the metrics
    
    for row in metrics_list:
        for k in ks:
            all_metrics[f'HR@{k}'].append(row[f'HR@{k}'])
            all_metrics[f'MRR@{k}'].append(row[f'MRR@{k}'])
            all_metrics[f'NDCG@{k}'].append(row[f'NDCG@{k}'])
            all_metrics[f'invalid_output@{k}'].append(row[f'invalid_output@{k}'])

    
    # Calculate the average for each metric and the invalid output count
    results = {
        'case': case,
        **{f'{metric}@{k}': np.mean(all_metrics[f'{metric}@{k}']) for k in ks for metric in ['HR', 'MRR', 'NDCG', 'invalid_output']},
    }

    logging.info("Results %s", results)

    with open(result_file, 'w') as f:
        logging.info("Save Metrics", )
        json.dump(results, f, indent=4)
        
    logging.info("Result saved to %s", result_file)
    logging.info("-"*40)

    wandb.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to .")
    parser.add_argument('--dir', type=str, help="The directory to the files.", default= r"./" )
    parser.add_argument('--dataset', type=str, help="The name of the dataset", choices=['ml-1m'], default='ml-1m') 
    parser.add_argument('--folder', type=str, help="The folder of the dataset", default="k_all_c_all")  
    parser.add_argument('--prompts', type=str, help="The path to the prompts", default="prompts.json")  
    parser.add_argument('--prompt_type', type=str, help="The type of the prompt", default="prompt_icl_ml")  
    parser.add_argument('--case', type=str, help="Whether cold or warm", choices=['cold', 'warm'], default='cold') 
    parser.add_argument('--model_id', type=str, help="The model name from Hf", default= "meta-llama/Llama-3.1-8B-Instruct") 
    parser.add_argument('--ks', type=int, help="A list of k topfor evaluation", default=[1]) 
    parser.add_argument('--seed', type=int, help="The seed for the run", default= 42) 
    parser.add_argument('--wandb_project', type=str, help="Weights and Biases project name.", default="")
    parser.add_argument('--wandb_run_name', type=str, help="Weights and Biases run name.", default="")
 
    args = parser.parse_args()
    main(args)


