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
As opposed to full fine-tuning, prefix tuning is a lightweight alternative which means freezing the whole (generative) transformer model and only updating weights of several vectors prepended to the input. When the model generates subsequent tokens, it will attend to those prepended tokens as well. We hope to learn a task description in this way, so that this small newly learnt matrix _softly prompts_ the model to generate more plausible outputs for our task as compared to generation from raw task data.
See [Li et al. 2021](https://arxiv.org/pdf/2101.00190) for more details. 

Our implementations:  
1. Our earliest implementation (class definitions [here](https://github.com/katja-kolos/foundation_models/blob/main/adapter_experiments/prefix_tuning_Param.py), script [here](https://github.com/katja-kolos/foundation_models/blob/main/adapter_experiments/perform_prefix_tuning_Param_experiment.py)) created an MLP network as a prefix of user-defined length (e.g. 10 tokens); those tokens were, however, not actual tokens from vocabulary, but initializations of the size of embeddings. This setup did alter the generations, making them shorter and more appropriate. We do not include it into our final report, though.
2. We later did a past key-value implementation (class definitions [here](https://github.com/katja-kolos/foundation_models/blob/main/adapter_experiments/prefix_tuning.py), script [here](https://github.com/katja-kolos/foundation_models/blob/main/adapter_experiments/perform_tuning_experiment.py), replicating original work from the above mentioned [Li et al. 2021](https://arxiv.org/pdf/2101.00190) paper's github (precisely, `get_prompt...` functions defined [here](https://github.com/XiangLi1999/PrefixTuning/blob/cleaned/gpt2/train_control.py), but a simplified version e.g. without dropout). 
3. We finally opted for using a [`PEFT`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning/model.py) implementation.

# How it runs
The notebooks were initially in the root of the project (allowing to import utils, e.g. from `helpers.py`. Remember to adjust PATH if launching from this directory. 
