This repository accompanies the paper LLMs For Warm and Cold Next-Item Recommendation: A Comparative Study across Zero-Shot Prompting, In-Context Learning and Fine-Tuning	

---

Included Files

The `datasets` folder includes the MovieLens data for the study after preprocesing

The `src` folder includes the preprocessing logic of the dataset as well as helpful utils functions

The `inference` folder includes the inference scripts of all used methods 

The `k-shot` folder includes the preprocessing logic for the sensitivity study (different sizes of training set)

The `train` file includes the logic for fine-tuning the LLM for the task 

The `prompts.json` file includes the crafted prompts used for zero-shot, in-context learning and fine-tuned models

The `metrics` file includes the logic for the evaluation

The `compute_avg_metrics` file includes a simple function to aggregate results of different independent runs 
