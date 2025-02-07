# What is here

This folder groups our notebooks and scripts for adapter learning experiments. 

# Experimental Setup

For those experiments, we took two pre-trained small multimodal LLMs and tried to adapt them to produce solutions for ScienceQA tasks.

The LLMs we tried were: 
- the smallest multimodal qwen2 instruct model: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- a multimodal paligemma model of comparable size (quantized version): [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)

We learned two types of adapters: 
- Prefix Tuning, and
- LoRA (see a good tutorial [here](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)).

We experienced with training on original ScienceQA dataset vs [outputs](https://github.com/katja-kolos/foundation_models/blob/main/gemini_1_5_flash_output_train.csv) of a [Gemini 1.5 Flash](https://ai.google.dev/competition/projects/multimodal-gemini-15-flash-api) model, that we chose as a champion after the [benchmarking](https://github.com/katja-kolos/foundation_models/tree/main/benchmarking_val) phase. 

# Prefix Tuning Details 
For prefix tuning, we first attempted an implementation of our own, directly working with past keys and values. 
An even earlier implementation (script [here](https://github.com/katja-kolos/foundation_models/blob/main/adapter_experiments/perform_prefix_tuning_Param_experiment.py)) created an MLP network as a prefix of user-defined length (e.g. 10 tokens); those tokens were, however, not actual tokens from vocabulary, but initializations of the size of embeddings. This setup did alter the generations, making them shorter and more appropriate. We do not include it into our final report, though, as we later did a PKV implementation, and finally opted for using a `PEFT` implementation. 

# How it runs
The notebooks were initially in the root of the project (allowing to import utils, e.g. from `helpers.py`. Remember to adjust PATH if launching from this directory. 
