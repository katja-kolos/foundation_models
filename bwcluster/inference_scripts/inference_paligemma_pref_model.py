# Script to download and inference with mistral models

import csv
import os
import pandas as pd
import torch

from collections import defaultdict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import logging
from datetime import datetime

# Generate timestamped filename
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"inference_qwen_pref_{start_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()            # Also log to console (optional)
    ]
)

#from vllm import LLM, TokensPrompt
#from vllm import EngineArgs, LLMEngine
#from vllm.sampling_params import SamplingParams
#from vllm.multimodal import MultiModalDataBuiltins

from huggingface_hub import login
from datasets import load_dataset

from PIL import Image

## UTILS: LOAD MODEL
# login to hf -- paligemma does not allow unauthorized downloads
def login_to_hf():
    try:
        with open("hf_token", "r") as f:
            token = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("HuggingFace token file not found. Please ensure 'hf_token' exists.")
        return

    login(token=token)

    logging.info("Login with token: done")

DEVICE = 'cuda'

# download or load from cache the model, put on GPU or distribute among GPUs, prepare CUDA graph
def prepare_model(model_name, local_adapter_path, checkpoint, max_model_len):

    import torch
    from peft import PeftModel, PeftConfig
    #from transformers import AutoModelForCausalLM

    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, #"auto"
            device_map="auto",
    ).to(DEVICE)
    logging.info('Loaded model')

    model.config.vocab_size = model.config._vocab_size
    model.config.hidden_size = model.config.hidden_size // 2

    #config = PeftConfig.from_pretrained("qwen_qwen2-vl-2b-instruct_golden/checkpoint-9500") #checkpoint-9500
    model = PeftModel.from_pretrained(
            model,
            f"{local_adapter_path}/checkpoint-{checkpoint}",
            #is_trainable=True # 👈 here
    ).to(DEVICE)

   # model.config.vocab_size = model.config._vocab_size
    #model.config.hidden_size = model.config.hidden_size // 2

    #llm = LLM(model=model_name, 
    #          tokenizer_mode="auto",
    #          task="generate",
    #          tensor_parallel_size=tensor_parallel_size,  # Can be adjusted for multi-GPU setup
    #          cpu_offload_gb=cpu_offload_gb,
    #          dtype="float16",
    #          # max_model_len=max_model_len 
    #         )

    logging.info("Load model with prefix: done")
    
    #return llm
    return model

## UTILS: LOAD DATA
def prepare_data(dataset_name="derek-thomas/ScienceQA", split="validation", resume_index=0):
    dataset = load_dataset(dataset_name)
    
    validation_data = dataset[split]

    logging.info(f"Original dataset has {len(validation_data)} datapoints >")
    
    filtered_validation_data = validation_data.filter(lambda example: example["lecture"] != "" and example["solution"] != "")
    
    logging.info(f"Filtered dataset has {len(filtered_validation_data)} datapoints >")

    filtered_validation_data = filtered_validation_data.shard(num_shards=1, index=0).select(range(resume_index,len(filtered_validation_data)))
    
    logging.info(f"Processing will start at index {resume_index}.")

    logging.info("Load data: done")

    return filtered_validation_data

## UTILS: SAVE RESULTS
# Helper function to save each processed result to the appropriate CSV
def save_to_csv(setting, model_name, entry, split, exp_setting):
    model_name = model_name.lower().replace("/", "_").replace(".", "_")
    
    results_save_path = f"./benchmarking/{model_name}"
    os.makedirs(results_save_path, exist_ok=True)

    file_path = f"{results_save_path}/{model_name}_pref{split}-{exp_setting}_output_setting_{setting}.csv"
    df = pd.DataFrame([entry])  # Single-row DataFrame for appending
    with open(file_path, 'ab') as f:
        df.to_csv(f, header=f.tell()==0, index=False, sep="\t", quoting=csv.QUOTE_ALL)  # Write header if file is empty

## UTILS: PROCESSING

setting_templates = {
    "QTCH": "Question: {question}\n Task: {task}\n Choices: {choice}\n Hint: {context}\n Instruction: {prompt_answer_and_solution}",
    "QTCHL": "Question: {question}\n Task: {task}\n Choices: {choice}\n Hint: {context} \nLecture: {lecture}\n Instruction: {prompt_answer_and_solution}",
    "QTCHLS": "Question: {question}\n Task: {task}\n Choices: {choice}\n Hint: {context} \nLecture: {lecture}\nSolution: {solution}\n Instruction: {prompt_answer_and_solution}",
    "QTCHS": "Question: {question}\n Task: {task}\n Choices: {choice}\n Hint: {context} \nSolution: {solution}\n Instruction: {prompt_answer_and_solution}",
}

#def process_datapoint_at_setting(example, idx, setting, llm, model_name, sampling_params, resumed_index_from):
def process_datapoint_at_setting(example, idx, setting, model, model_name, processor, tokenizer, resumed_index_from, split, exp_setting):
    # vllm is able to process image urls and raw image data (docs: https://github.com/vllm-project/vllm/blob/main/docs/source/design/input_processing/input_processing_pipeline.rst#id16)
    image = example["image"]
    subject = example["subject"]
    prompt_answer_and_solution = '\nPlease output the answer in JSON style with an answer and a solution field'

    text_input = setting_templates[setting].format(
        question=example["question"], 
        task=example["task"], 
        choice=example["choices"], 
        context=example["hint"], 
        lecture=example["lecture"], 
        solution=example["solution"], 
        prompt_answer_and_solution=prompt_answer_and_solution)
  
    #processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    #create empty image if none is given
    if not image:
        image = Image.new("RGB", (224, 224), (0, 0, 0))
        
    # Simulate chat interaction
    #messages = [
    #    {
    #        "role": "user",
    #        "content": [
    #            {
    #                "type": "image",
    #                "image": image,
    #            },
    #            {"type": "text", "text": text_input},
    #        ],
    #    }
    #]
    
    # Preparation for inference
    #text = processor.apply_chat_template(
    #    messages, tokenize=False, add_generation_prompt=True
    #)

    text = text_input
    
    logging.info(text)
   
    #images = [image.resize((224, 224)) for image in images]
    images = [image.resize((224, 224))]
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        #padding="longest"
    ).to(DEVICE).to(dtype=torch.bfloat16)
    logging.info("prepared inputs")

    with torch.no_grad():
        outputs = model(**inputs)
        #logging.info("got outputs")
        output_ids = outputs.logits.argmax(-1)[0]
        #logging.info(output_ids)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        #logging.info(output_text)
        #logging.info("Nice!")

    #outputs = llm.generate({
    #    "prompt": text,
    #    "multi_modal_data": {
    #        "image": [image]
    #    },
    #}, sampling_params=sampling_params)    

    #answer = ''
    #for o in outputs:
    #    answer += o.outputs[0].text #normally expecting one output in 1 by 1 processing

    #if answer == '':
    #    logging.warn(f"Empty answer for: idx={idx} setting={setting}")
    #    answer = "NA"

    entry = {"idx": resumed_index_from+idx, "input": text_input, "output": output_text, "subject": subject}
    save_to_csv(setting, model_name, entry, split, exp_setting)

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/paligemma2-3b-pt-224",
        help="The name of the model to download and use as base. Default: google/paligemma2-3b-pt-224",
    )
#    parser.add_argument(
#        "--ngpus",
#        type=int,
#        default=1,
#        help="n > 0 of GPUs available for inference",
#    )

#    parser.add_argument(
#        "--cpuRAM",
#        type=int,
#        default=0,
#        help="How much CPU RAM can be used to unload parts of the model that do not fit into GPU memory",
#    )

    parser.add_argument(
        "--maxtokens",
        type=int,
        default=2048,
        help="max_tokens of max_model_len at LLM initialization. Default: 2048",
    )

    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="Task setting to process. One of: QTCH, QTCHL, QTCHLS, QTCHS"
    )

    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Index of datapoint to resume processing from. Default: 0"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="derek-thomas/ScienceQA",
        help='HF dataset to be scored with the model. Default: "derek-thomas/ScienceQA"'
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split of dataset to be scored with the model. Default: validation"
    )

    parser.add_argument(
        "--local_path",
        type=str,
        default="google_paligemma2-3b-pt-224_golden",
        help="Path to adapter model saved locally"
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=57580,
        help="checkpoint int; default 57580"
    )

    args = parser.parse_args()

    # Log in to huggingface 
    login_to_hf() #not necessary for qwen
    
    # Load and prepare data
    data = prepare_data(dataset_name=args.dataset, split=args.split, resume_index=args.resume)

    # Load and inference the model
    #"qwen_qwen2-vl-2b-instruct_golden/checkpoint-9500"
    llm = prepare_model(args.model_name, args.local_path, args.checkpoint,  max_model_len=args.maxtokens)
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #sampling_params = SamplingParams(
    #    max_tokens=args.maxtokens, #max_new_tokens=128 in model card
    #    stop=["}\n"],
    #    include_stop_str_in_output=True,
    #)
    data.map(lambda example, idx: process_datapoint_at_setting(
        example, idx, 
        args.setting, 
        llm, args.model_name, 
        processor, tokenizer,
        #sampling_params=sampling_params,
        resumed_index_from=args.resume,
        split=args.split,
        exp_setting=args.local_path), with_indices=True)

if __name__ == "__main__":
    main()
