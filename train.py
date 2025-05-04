import os
import torch
import random
import wandb
import argparse
import torch
import logging
from typing import List
from rapidfuzz import process, fuzz
from datasets import load_dataset
import re
import difflib
import bitsandbytes as bnb
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import json
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
    pipeline
)

from peft import (
    LoraConfig,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType

)

from trl import SFTTrainer, SFTConfig,  DataCollatorForCompletionOnlyLM


import time
# Environment setup
os.environ['LIBRARY_PATH'] = './'



# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),  
    #logging.FileHandler('./logs/ML-1M/train.log') 
    ]
)


logging.info("Script is starting...") 

def main(args):

    # Model and training parameters
    base_model = args.model_id
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_dir = args.output_dir
    seed = args.seed

    prompts_path = args.prompts
    prompt_type = args.prompt_type

    # Training hyperparameters
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    cutoff_len = args.cutoff_len

    # Lora hyperparameters
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = args.lora_target_modules

    # LLM hyperparameters
    train_on_inputs = args.train_on_inputs
    group_by_length = args.group_by_length

    # wandb params
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_watch = args.wandb_watch
    wandb_log_model = args.wandb_log_model
    
    resume_from_checkpoint = args.resume_from_checkpoint

    logging.info('Configs: %s', args)
    
    set_seed(seed)

    logging.info(
        f"Training model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    assert base_model, "Please specify a --base_model, e.g. --base_model='meta-llama/Meta-Llama-3-8B'"

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
    

    def apply_permute_sample(example):
        example['permuted_candidates'] = permute_sample(example['parsed_candidates'], seed=None) 
        return example
    
    def to_string(permute_sample):
        return ', '.join([f'"{candidate}"' for candidate in permute_sample])
    

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
    def formatting_prompts_func_movielens(examples, instruction):
        histories = examples["history"]
        candidates = examples["permuted_candidates_string"]
        outputs = examples["gold"]
        messages = []

        for history, candidate, output in zip(histories, candidates, outputs):
            user_content = instruction.format(history, candidate)
            assistant_content = f"{output}"

            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for recommendations."
                },
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ]
            messages.append(chat)
            
        return { "messages" : messages }
    


    def formatting_prompts_func_bookcrossing(examples, instruction):
        histories = examples["history"]
        candidates = examples["permuted_candidates_string"]
        outputs = examples["gold"]
        messages = []

        for history, candidate, output in zip(histories, candidates, outputs):
            user_content = instruction.format(history, candidate)
            assistant_content = f"{output}"  
            
            chat = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for recommendations."
                },
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ]
            messages.append(chat)
        return { "messages" : messages }


    def formatting_prompts_func_for_test(examples, instruction):
        inputs = examples["history"]
        candidates = examples["permuted_candidates_string"]
        messages = []
        for input, candidate in zip(inputs, candidates):
            user_content = instruction.format(input, candidate)
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
    


    torch.cuda.empty_cache()
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=wandb_run_name  
        )

    logging.info(f"WandB Run Name: {wandb_run_name}")

    #model
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logging.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    #model & tokenizer
    PAD_TOKEN= "<|pad|>"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True) 
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    tokenizer.padding_side = "right" 
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map = "auto",
        quantization_config=bnb_config,
    )

    
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of =  8)
    print("tokenizer.bos_token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("tokenizer.eos_token", tokenizer.eos_token, tokenizer.eos_token_id)
    print("tokenizer.pad_token", tokenizer.pad_token, tokenizer.pad_token_id)
    tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    


    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_weights.bin"
        )
        if not os.path.exists(checkpoint_name):
            resume_from_checkpoint = False  

        if os.path.exists(checkpoint_name):
            logging.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logging.info(f"Checkpoint {checkpoint_name} not found")

   
   #Load Prpmpts
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)


    #Data
    if train_data_path.endswith(".json"):
        train_data = load_dataset("json", data_files= train_data_path)
    if val_data_path.endswith(".json"):
        val_data = load_dataset("json", data_files=val_data_path)
    

    #loading
    train_data["train"] = train_data["train"].shuffle(seed=None)    
    val_data["train"] = val_data["train"].shuffle(seed=None)


    #parsing
    train_data = train_data.map(apply_parse_candidates)
    val_data = val_data.map(apply_parse_candidates)

    #permute
    train_data = train_data.map(lambda example: apply_permute_sample(example))
    train_data = train_data.map(apply_to_string)

    val_data = val_data.map(lambda example: apply_permute_sample(example))
    val_data = val_data.map(apply_to_string)

    
    #prompting
    instruction = prompts[prompt_type]


    if train_data_path.split('/')[1].startswith('ml'):             
        train_data = train_data["train"].map(
            lambda examples: formatting_prompts_func_movielens(examples, instruction), 
            batched=True)
    else:
        train_data = train_data["train"].map(
            lambda examples: formatting_prompts_func_bookcrossing(examples, instruction),
            batched = True)

    val_data = val_data["train"].map(
            lambda examples: formatting_prompts_func_bookcrossing(examples, instruction),
            batched = True)

    #chat template
    train_data = train_data.map(lambda x: {"tokenized_messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
    val_data = val_data.map(lambda x: {"tokenized_messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)})

    #gold rank
    train_data = train_data.map(calculate_gold_ranks)


    #Tesing the original model
    print('train_data example')
    pprint(train_data[0])
    print("-"*20)



    pipe = pipeline(task="text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    max_new_tokens = 200,
                    return_full_text = False)
    
 
    outputs = pipe(val_data[0]['tokenized_messages'])
    response = f"""
    tokenized_messages: {val_data[0]['tokenized_messages']}
    gold = {val_data[0]['gold']}
    prediction = {outputs[0]['generated_text']}
    """

    print('Testing original Model', response)
        
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model) 
    model = get_peft_model(model, lora_config)
    logging.info('Model:')
    print_trainable_parameters(model)
    print('model.config:', model.config)

   # Train on completion only
    response_template = "<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer= tokenizer)


    #example
    examples= [ train_data[0]['tokenized_messages']]
    encodings = [tokenizer(e) for e in examples]
    dataloader = DataLoader(encodings, collate_fn = collator, batch_size= 1)
    batch = next(iter(dataloader))



    steps = 1000 
    model.gradient_checkpointing_enable()
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator= collator,
        train_dataset = train_data,
        eval_dataset = val_data,
        args = SFTConfig(
            output_dir=output_dir,
            dataset_text_field="tokenized_messages",
            max_seq_length=cutoff_len,
            seed=seed,
            push_to_hub=False,
            hub_private_repo=True,
            hub_strategy='checkpoint',
            disable_tqdm= False,
            num_train_epochs = num_epochs,
            per_device_train_batch_size = micro_batch_size,
            per_device_eval_batch_size = micro_batch_size,
            warmup_ratio= 0.1,
            learning_rate=learning_rate,
            fp16=True,
            optim = "paged_adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "constant",
            logging_steps = 50,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps= steps,
            save_steps= steps, 
            gradient_checkpointing = True,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to= "wandb" if use_wandb else None,
            run_name= wandb_run_name if use_wandb else None, 
            dataset_kwargs = {
                "add_special_tokens" : False,
                "append_concat_token": False
            }
        ),
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    model.config.use_cache = False

    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    end_time = time.time()
    logging.info("-"*30)
    logging.info("Time taken to train: %.2f seconds", end_time - start_time)
    logging.info("-"*30)


    if use_wandb:
        wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to fine-tune a model.")
    parser.add_argument('--train_data_path', type=str, help="The path to the training data.", default="datasets/ml-1m/k_all_c_all/train.json")
    parser.add_argument('--val_data_path', type=str, help="The path to the validation data.",  default="datasets/ml-1m/k_all_c_all/valid.json")
    parser.add_argument('--output_dir', type=str, help="The output directory for model checkpoints.",  default= "llm_models/ml-1m/ml_k_all_c_all") 
    parser.add_argument('--model_id', type=str, help="The model name from Huggingface.",  default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--batch_size', type=int, help="Batch size.", default=1)
    parser.add_argument('--micro_batch_size', type=int, help="Micro batch size.", default=1)
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs.", default=3)
    parser.add_argument('--learning_rate', type=float, help="Learning rate.", default=1e-4)
    parser.add_argument('--cutoff_len', type=int, help="Maximum sequence length.", default=1024)
    parser.add_argument('--lora_r', type=int, help="Lora r parameter.", default= 32)
    parser.add_argument('--lora_alpha', type=int, help="Lora alpha parameter.", default=16)
    parser.add_argument('--lora_dropout', type=float, help="Lora dropout parameter.", default=0.05)
    parser.add_argument('--lora_target_modules', type=List[str], help="Lora target modules.", default=["self_attn.q_proj", "self_attn.k_proj","self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    parser.add_argument('--train_on_inputs', type=bool, help="Whether to train on inputs.", default=False)
    parser.add_argument('--group_by_length', type=bool, help="Whether to group by length.", default=False)
    parser.add_argument('--wandb_project', type=str, help="Weights and Biases project name.", default="")
    parser.add_argument('--wandb_run_name', type=str, help="Weights and Biases run name.", default="")
    parser.add_argument('--wandb_watch', type=str, help="Weights and Biases watch parameter.", default="all")
    parser.add_argument('--wandb_log_model', type=str, help="Weights and Biases log model parameter.", default='')
    parser.add_argument('--resume_from_checkpoint', type=str, help="Resume from checkpoint.", default=None)
    parser.add_argument('--prompts', type=str, help="The path to the prompts", default="prompts.json")  
    parser.add_argument('--prompt_type', type=str, help="The type of the prompt", default="prompt_instruct_ml")  
    parser.add_argument('--seed', type=int, help="The seed for the run", default= 42) 
    args = parser.parse_args()
    main(args)



