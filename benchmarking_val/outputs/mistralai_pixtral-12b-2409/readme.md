# ReadMe

This output was generated on BWUniCluster in several runs, partly using `dev_gpu_4_a100` queue and partly using `gpu_4` queue. 

Each job was using a pixtral model served by `vllm` on 4 gpus. 

Warning: originally, there were duplicate rows in raw outputs, which were manually removed. As the temperature was non-zero,  generations were different for the same task (different explanations, and even different answers).

Warning 2: the generations may continue beyond a stop token, degenerating into complete nonsense (code, crazy text, gibberish). A postprocessing script will be added to cut answers of the model when the first stop signal occurs (`<\s>` or `[STOP]`, maybe other tokens). Those trailing generations were removed by a manually designed post-processing script.

The model used is `mistralai/Pixtral-12B-2409` (note: from official huggingface, and not the community edition).

The output format currently in the answer field is not the number of the correct answer but the correct answer as a string. 

