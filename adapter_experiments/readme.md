# What is here

This folder groups our notebooks and scripts for adapter learning experiments. 

# Experimental Setup

For those experiments, we took two pre-trained small multimodal LLMS and tried to adapt them to produce solutions for ScienceQA tasks.

The LLMs we tried were: 
- the smallest multimodal qwen2 instruct model: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- a multimodal paligemma model of comparable size (quantized version): [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)

We learned two types of adapters: 
- Prefix Tuning, and
- LoRA (see a good tutorial [here](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)).

We experienced with training on original ScienceQA dataset vs outputs of a Gemini model that we chose as a champion after the benchmarking phase. 

# Prefix Tuning Details 
For prefix tuning, we first attempted an implementation of our own, directly working with attention keys and values. 
An even earlier implementation was a _Param_ version.
We finally opted for using a `PEFT` implementation. 

# How it runs
The notebooks were initially in the root of the project (allowing to import utils, e.g. from `helpers.py`. Remember to adjust PATH if launching from this directory. 
