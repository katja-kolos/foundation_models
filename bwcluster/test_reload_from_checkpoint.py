import torch
from peft import PeftModel, PeftConfig
#from transformers import AutoModelForCausalLM

from transformers import AutoModelForImageTextToText, Qwen2VLForConditionalGeneration
print('Hi there!')
model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16, #"auto"
            device_map="auto",
)
print('Loaded model')

config = PeftConfig.from_pretrained("qwen_qwen2-vl-2b-instruct_golden/checkpoint-9500") #checkpoint-9500
print('Loaded config')
print(config)
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
model = PeftModel.from_pretrained(model, 
    "qwen_qwen2-vl-2b-instruct_golden/checkpoint-9500",
    #is_trainable=True # 👈 here
    )
# check if it's working
#model.print_trainable_parameters()
#model()
