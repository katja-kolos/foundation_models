import argparse
import gc
import logging
import numpy as np
import pandas as pd
import torch
import warnings

from datasets import load_dataset
from datetime import datetime
from peft import PrefixTuningConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTConfig


from qwen_vl_utils import process_vision_info #this is some basic if-else image processing, can be reused as is for paligemma runs
from prefix_tuning import PrefixTuningModelPastKeyValues, PrefixDataset, prefix_collate
from helpers import *


"""
Script to prefix-train a generative MLLM
Was initially written for qwen, current adaptation is for paligemma
The training data is the multimodal ScienceQA dataset. 
The targets are free-text solutions, which are compared to solutions generated by the model. The answer field (int: [1:4]) is not used in this implementation.
Assumed model architecture: autoregressive textual model + visual encoder
Script only runs on cuda and does not support CPU-only runs
"""
warnings.filterwarnings("ignore", message="You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special image tokens in the text, as many tokens as there are images per each text. It is recommended to add `<image>` tokens in the very beginning of your text and `<bos>` token after that. For this call, we will infer how many images each text has and add special tokens.")

# Empty cache just in case
torch.cuda.empty_cache()

# Logging
# Generate timestamped filename
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"prefix-tuning_{start_time}.log"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()            # Also log to console (optional)
    ]
)

device = 'cuda' # torch.cuda.get_device_name(0)
logging.info(torch.cuda.get_device_name(0))
logging.info('Memory Usage:')
logging.info(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
logging.info(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')


def get_question_text(problem):
    question = problem['question']
    return question


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def build_prompt(question_data, use_lecture=False, use_solution=False):
    question = get_question_text(question_data)
    choices = get_choice_text(question_data, [choice_num for choice_num in range(5)])
    hint = get_context_text(question_data, False)
    task = question_data['task']
    input_prompt = f'Question: {question}\n Task: {task}\n Choices: {choices}\n Hint: {hint}'
    if use_lecture:
        lecture = f'\n Lecture: {question_data["lecture"]}'
        input_prompt += lecture
    if use_solution and question_data["solution"]:
        solution = f'\n Solution: {question_data["solution"]}'
        input_prompt += solution
    return input_prompt

def build_message(row):
    row_input = build_prompt(row)
    image = row['image'] if row['image'] else Image.new("RGB", (224, 224), (0, 0, 0))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": row_input },
            ],
        }
    ]
    return messages

def build_message_teacher(row):
    image = row['image'] if row['image'] else Image.new("RGB", (224, 224), (0, 0, 0))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": row["input"] }, #teacher outputs already have an input column made by build_prompt previously
            ],
        }
    ]
    return messages


def prepare_datasets(exp_setting: str):
    dataset_id = "derek-thomas/ScienceQA"
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train", "validation", "test"])
    
    #filters
    train_dataset = train_dataset.filter(lambda example: example['solution']!="")
    eval_dataset = eval_dataset.filter(lambda example: example['solution']!="")
    #test_dataset = test_dataset.filter(lambda example: (example['solution']!="") & (example['lecture']!=""))

    if exp_setting == 'golden':
        return train_dataset, eval_dataset
    elif exp_setting == 'teacher':
        # data from Gemini for KD
        train_dataset_gemini = pd.read_csv('gemini_1_5_flash_output_train.csv', sep="\t")[['index', 'input', 'answer', 'explanation']]
        train_dataset_df = pd.DataFrame(train_dataset).reset_index()
        train_dataset_gemini = pd.merge(train_dataset_gemini, train_dataset_df[['index', 'image', 'solution']], on='index')
        train_dataset_gemini["solution"] = train_dataset_gemini["explanation"]
        return train_dataset_gemini, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Select training experiment setting: what will be used as training data") 
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="golden",
        help="'golden' for dataset, 'teacher' for gemini outputs. Default: 'golden'"
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="n gpus to use at training. Default: 1",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch_size for both train and validation dataloaders"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="num_epochs; default: 10"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="exact hf path to model; default: 'Qwen/Qwen2-VL-2B-Instruct'"
    )

    args = parser.parse_args()
    
    model_name = args.model_name
    logging.info(f'Model: {model_name}')
    
    train_dataset, eval_dataset = prepare_datasets(exp_setting=args.exp_setting)

    if model_name == "Qwen/Qwen2-VL-2B-Instruct":
        from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, #"auto"
            device_map="auto",
        )
        logging.info('Loaded model')
    
        processor = AutoProcessor.from_pretrained(model_name)
        logging.info('Loaded processor')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info('Loaded tokenizer')
        
        def collate_fn_qwen(examples):
            # Get the texts and images, and apply the chat template
            texts = [
                processor.apply_chat_template(example, tokenize=False) for (example,_) in examples
            ]  # Prepare texts for processing
            image_inputs = [process_vision_info(example)[0] for (example,_) in examples]  # Process the images to extract inputs

            # Tokenize the texts and process the images
            batch = processor(
                text=texts, images=image_inputs, return_tensors="pt", padding=True
            )
            max_length = batch["input_ids"].size(1)
            example_labels = [label for (x, label) in examples]
            labels = tokenizer(example_labels, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
            batch["labels"] = labels  # Add labels to the batch
            return batch  # Return the prepared batch


        collate_fn = collate_fn_qwen

        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=50,
        )

        # Configure training arguments
        training_args = SFTConfig(
            output_dir=f"{args.model_name.replace('/', '_').lower()}_{args.exp_setting}",  # Directory to save the model
            num_train_epochs=args.num_epochs,  # Number of training epochs
            per_device_train_batch_size=args.batch_size,  # Batch size for training
            per_device_eval_batch_size=args.batch_size,  # Batch size for evaluation
            #gradient_accumulation_steps=8,  # Steps to accumulate gradients
            #gradient_checkpointing=False,  # Important to be False!!! Will not work with True
            # Optimizer and scheduler settings
            #optim="adamw_torch_fused",  # Optimizer type
            #learning_rate=5e-5,  # Learning rate for training
            #lr_scheduler_type="constant",  # Type of learning rate scheduler
            # Logging and evaluation
            #logging_steps=10,  # Steps interval for logging
            #eval_steps=10,  # Steps interval for evaluation
            #eval_strategy="steps",  # Strategy for evaluation
            #save_strategy="steps",  # Strategy for saving the model
            #save_steps=20,  # Steps interval for saving
            #metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            #greater_is_better=False,  # Whether higher metric values are better
            #load_best_model_at_end=True,  # Load the best model after training
            # Mixed precision and gradient settings
            #bf16=True,  # Use bfloat16 precision
            #tf32=True,  # Use TensorFloat-32 precision
            #max_grad_norm=0.3,  # Maximum norm for gradient clipping
            #warmup_ratio=0.03,  # Ratio of total steps for warmup
            # Hub and reporting
            push_to_hub=False,  # Whether to push model to Hugging Face Hub
            report_to="none", #"wandb",  # Reporting tool for tracking metrics
            # Gradient checkpointing settings
            # Dataset configuration
            dataset_text_field="",  # Text field in dataset
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            # max_seq_length=1024  # Maximum sequence length for input
            #padding_side='right',
        )

        training_args.remove_unused_columns = False  # Keep unused columns in dataset
        
        # Apply PEFT model adaptation
        peft_model = get_peft_model(model, peft_config)

        if args.exp_setting == "teacher":
            train_dataset = [(build_message_teacher(sample[1]), sample[1]["solution"]) for sample in train_dataset.iterrows()]
        else:
            train_dataset = [(build_message(sample), sample["solution"]) for sample in train_dataset]
        eval_dataset = [(build_message(sample), sample["solution"]) for sample in eval_dataset]

    elif model_name == "google/paligemma2-3b-pt-224":
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        from helpers import login_to_hf
        login_to_hf()
        logging.info("Login with token: done")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
        )
        logging.info('Loaded model')
        

        model.config.vocab_size = model.config._vocab_size
        model.config.hidden_size = model.config.hidden_size // 2
        
        logging.info(f'Set model.config.vocab_size to {model.config.vocab_size}')
        logging.info(f'Set model.config.hidden_size to {model.config.hidden_size}')

        processor = PaliGemmaProcessor.from_pretrained(model_name)
        logging.info('Loaded processor')
        tokenizer = processor.tokenizer #AutoTokenizer.from_pretrained(model_name)
        logging.info('Loaded tokenizer')
        
        def collate_fn_paligemma(examples):
            texts = [text for (text, image, label) in examples]
            #image_inputs = [image.resize((224, 224)) for (text, image, label) in examples]
            image_inputs = [image.resize((224, 224)) if image else Image.new("RGB", (224, 224), (0, 0, 0)) for (text, image, label) in examples]

            # Tokenize the texts and process the images
            batch = processor(
                text=texts, images=image_inputs, return_tensors="pt", padding=True
            ).to(dtype=torch.bfloat16)
            max_length = batch["input_ids"].size(1)
            example_labels = [label for (text, image, label) in examples]
            labels = processor.tokenizer(example_labels, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
            batch["labels"] = labels  # Add labels to the batch
            return batch  # Return the prepared batch

        collate_fn = collate_fn_paligemma

        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=50,
            num_attention_heads=4, #model.config.text_config? 
        )
        
        # Configure training arguments
        training_args = SFTConfig(
            output_dir=f"{args.model_name.replace('/', '_').lower()}_{args.exp_setting}",  # Directory to save the model
            num_train_epochs=args.num_epochs,  # Number of training epochs
            per_device_train_batch_size=args.batch_size,  # Batch size for training
            per_device_eval_batch_size=args.batch_size,  # Batch size for evaluation
            #gradient_accumulation_steps=8,  # Steps to accumulate gradients
            #gradient_checkpointing=False,  # Important to be False!!! Will not work with True
            # Optimizer and scheduler settings
            #optim="adamw_torch_fused",  # Optimizer type
            #learning_rate=5e-5,  # Learning rate for training
            #lr_scheduler_type="constant",  # Type of learning rate scheduler
            # Logging and evaluation
            #logging_steps=10,  # Steps interval for logging
            #eval_steps=10,  # Steps interval for evaluation
            #eval_strategy="steps",  # Strategy for evaluation
            #save_strategy="steps",  # Strategy for saving the model
            #save_steps=20,  # Steps interval for saving
            #metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            #greater_is_better=False,  # Whether higher metric values are better
            #load_best_model_at_end=True,  # Load the best model after training
            # Mixed precision and gradient settings
            #bf16=True,  # Use bfloat16 precision
            #tf32=True,  # Use TensorFloat-32 precision
            #max_grad_norm=0.3,  # Maximum norm for gradient clipping
            #warmup_ratio=0.03,  # Ratio of total steps for warmup
            # Hub and reporting
            push_to_hub=False,  # Whether to push model to Hugging Face Hub
            report_to="none", #"wandb",  # Reporting tool for tracking metrics
            # Gradient checkpointing settings
            # Dataset configuration
            dataset_text_field="",  # Text field in dataset
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            # max_seq_length=1024  # Maximum sequence length for input
            #padding_side='right',
        )

        training_args.remove_unused_columns = False  # Keep unused columns in dataset

        # Apply PEFT model adaptation
        peft_model = get_peft_model(model, peft_config)

        if args.exp_setting == "teacher":
            train_dataset = [(sample[1]["input"], sample[1]["image"], sample[1]["solution"]) for sample in train_dataset.iterrows()] # sample["input"] is the output of build_prompt
        else:
            train_dataset = [(build_prompt(sample), sample["image"], sample["solution"]) for sample in train_dataset]

    logging.info('Starting training')
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    #logging.info('Saving assets')
    #save_assets(model, model_name, train_errors, val_errors, args.batch_size, adapter_type='prefix', exp_setting=args.exp_setting, num_epochs=NUM_EPOCHS_FT, saving_mode='adapter_only')
    saving_path = f"{args.model_name.replace('/', '_').lower()}_{args.exp_setting}_full_model"
    torch.save(model, saving_path)
    logging.info(f'Saved model to path: {saving_path}')
    #logging.info('Done')

if __name__ == "__main__":
    main()
