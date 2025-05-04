
import os
import sys
import random
import re
import json
import difflib
import numpy as np
import logging
import argparse
from collections import defaultdict
import torch
from rapidfuzz import process, fuzz
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, AutoModelForCausalLM

from llm_for_nir.src.utils import check_path
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

logging.info("-"*30)
logging.info("Script is starting...") 


def main(args):
        
    dir = args.dir
    dataset = args.dataset 
    folder = args.folder
    prompts_path = args.prompts
    case=args.case

    hf_token = args.hf_token

    model_id = args.model_id
    ks = args.ks 
    prompt_type = args.prompt_type
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

    login(token=hf_token, add_to_git_credential=False)


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
    
    def apply_permute_sample(example, seed):
            example['permuted_candidates'] = permute_sample(example['parsed_candidates'], seed=seed)
            return example


    def to_string(permuted_sample):
        return ', '.join([f'"{candidate}"' for candidate in permuted_sample])

    def apply_to_string(example):
        example['permuted_candidates_string'] = to_string(example['permuted_candidates'])
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


    #prompting
    def formatting_prompts_func_bookcrossing(examples, instruction):

        histories = examples["history"]
        candidates = examples["permuted_candidates_string"]
        messages = []
        for history, candidate in zip(histories, candidates):
            user_content = instruction.format(history, candidate)
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
        
        histories = examples["history"]
        candidates = examples["permuted_candidates_string"]
        messages = []

        for history, candidate in zip(histories, candidates): 
            user_content = instruction.format(history, candidate)

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            inputs = tokenizer(
                example['tokenized_messages'],
                return_tensors="pt",
                max_length=4096,
                truncation=True,
                ).to(device)

            with torch.amp.autocast("cuda"):
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
        candidates = example['permuted_candidates']  
        
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
        example['pred_rank'] = generate_rank_tensor(example['parsed_ranking'], example['permuted_candidates']).tolist()
        return example 
    

    #evaluation 
    def eval_hf_ds(example):
        metrics = {}
        pred_rank = torch.tensor(example['pred_rank'])  
        gold_rank = torch.tensor(example['gold_rank']) 
        metrics.update(eval(rank = pred_rank, labels =gold_rank, ks = ks))
        example['metrics'] = metrics
        return example

    #count hallucinated answers
    def count_invalid_outputs(example):
        parsed_ranking = example['parsed_ranking']
        permuted_candidate = example['permuted_candidates']
        
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


    #Test & Results path
    test_data_path = os.path.join(os.environ.get("PATH"), os.path.join('datasets', dataset, folder)) 
    test_file = os.path.join(test_data_path , 'test.json')

    if not os.path.exists(test_file):
        print(f"Test file {test_file} does not exist")
        sys.exit(1)

    result_path = os.path.join(os.environ.get("PATH"), os.path.join('results', dataset, 'zeroshot'))
    result_dir = os.path.join(result_path , folder)
    check_path(result_dir)

    result_file = os.path.join(result_dir, prompt_type + '_' + case + '_results' + '_seed_' + str(seed) + '.json') 
    
    logging.info("Test File %s", test_file)
    logging.info("Result File %s", result_file)
    

    #Start
    if test_file.endswith(".json"):
        test_data = load_dataset("json", data_files= test_file)
    
        #filter cold/warm
        if case != None:
            test_data['train'] = test_data['train'].filter(filter_function)
        

        #shuffle
        test_data["train"] = test_data["train"].shuffle(seed=seed)

        #parse candidates
        test_data = test_data.map(apply_parse_candidates)

        #permute
        test_data = test_data.map(lambda example: apply_permute_sample(example, seed=seed))
        
        #parse permuted candidate for llm (to string)
        test_data = test_data.map(apply_to_string)

        #gold rank
        test_data = test_data.map(calculate_gold_ranks)

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

        #pred rank
        test_data = test_data.map(generate_rank_tensor_hf_ds)

        logging.info("Start evaluation")

        #eval
        test_data = test_data.map(eval_hf_ds)

        #invalid output
        test_data = test_data.map(count_invalid_outputs)
        
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
        logging.info("-"*30)

        wandb.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to .")
    parser.add_argument('--dir', type=str, help="The directory to the files.", default= r"./" )
    parser.add_argument('--dataset', type=str, help="The name of the dataset", choices=["ml-1m"], default="ml-1m") 
    parser.add_argument('--folder', type=str, help="The folder of the dataset", default="k_all_c_all")  
    parser.add_argument('--prompts', type=str, help="The path to the prompts", default="prompts.json")  
    parser.add_argument('--prompt_type', type=str, help="The type of the prompt", default="prompt_zeroshot_ml")  
    parser.add_argument('--case', type=str, help="Whether cold or warm", choices=["cold", "warm"], default="warm") 
    parser.add_argument('--model_id', type=str, help="The model name from Hf", default= "meta-llama/Llama-3.1-8B-Instruct") 
    parser.add_argument('--ks', type=int, help="A list of k top for evaluation", default=[1]) 
    parser.add_argument('--seed', type=int, help="The seed for the run", default= 42) 
    parser.add_argument('--wandb_project', type=str, help="Weights and Biases project name.", default="")
    parser.add_argument('--wandb_run_name', type=str, help="Weights and Biases run name.", default="")
    args = parser.parse_args()
    main(args)


