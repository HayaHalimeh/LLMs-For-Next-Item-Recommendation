import os
import sys
import logging
import argparse
from datasets import load_dataset,  concatenate_datasets, DatasetDict
import json
from llm_for_nir.src.utils import check_path

import random
import numpy as np


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
    file_name = args.file_name 
    seed= args.seed
    smaples_nr = args.smaples_nr 


    logging.info('Configs: %s', args)

    os.environ["PATH"] = dir
    logging.info(f"Current directory set to: {dir}")

    # Paths
    data_path = os.path.join(os.environ.get("PATH"), os.path.join('datasets', dataset, folder))
    result_path = os.path.join(os.environ.get("PATH"), os.path.join('k-shot', dataset))


    # Train data file
    data_file = os.path.join(data_path, file_name)
    if not os.path.exists(data_file):
        print("train_data_path does not exist")
        sys.exit(1)


    # Result directory
    result_dir = os.path.join(result_path, str(smaples_nr))
    check_path(result_dir)

    # Result file
    result_file = os.path.join(result_dir, f"{file_name}")
    logging.info(" File %s", data_file)
    logging.info("Result File %s", result_file)

    # Load dataset
    if data_file.endswith(".json"):
        data = load_dataset("json", data_files= data_file)


    data_cold = data.filter(lambda example: example['case'] == 'cold')
    data_warm= data.filter(lambda example: example['case'] == 'warm')
        

    # Select k rows
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Select k rows for cold cases
    sampled_cold = data_cold['train'].shuffle(seed=seed).select(
        range(min(int(smaples_nr / 2), len(data_cold['train'])))
    )
    sampled_warm = data_warm['train'].shuffle(seed=seed).select(
        range(min(int(smaples_nr / 2), len(data_warm['train'])))
    )
    combined_samples = concatenate_datasets([sampled_cold, sampled_warm])


    # Convert to a DatasetDict if necessary
    combined_samples = DatasetDict({"train": combined_samples})
    combined_samples = combined_samples['train'].shuffle(seed=seed)  

    formatted_data = []
    for i in range(len(combined_samples)):
        formatted_data.append({
            "user_id": combined_samples["user_id"][i],
            "history": combined_samples["history"][i],
            "candidates": combined_samples["candidates"][i],
            "gold": combined_samples["gold"][i],
            "case": combined_samples["case"][i]
        })

    logging.info(combined_samples)
    # Save the reduced dataset
    with open(result_file, "w") as f:
        json.dump(formatted_data, f, indent=4)

    logging.info(f"Reduced dataset saved to: {result_file}")

    logging.info("-"*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to prepare the datasets.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed value')  
    parser.add_argument('--dir', type=str, help="The directory to the files.", default="./")
    parser.add_argument('--dataset', type=str, help="The dataset name.", default="ml-1m")
    parser.add_argument('--folder', type=str, help="The folder of the dataset", default="k_all_c_all")  
    parser.add_argument('--file_name', type=str, help="The name of the file", default="train.json")  
    parser.add_argument('--smaples_nr', type=int, help="The number of few shots", default=128) 


    args = parser.parse_args()
    main(args)


