# ReadMe

Note: This output was generated on BWCluster in several runs, partly using dev_gpu_4_a100 queue and partly using gpu_4 queue. The output as of now is incomplete: still waiting for the end of QTCHLS and QTCHL jobs. 

Each job was using a pixtral model served by `vllm` on 4 gpus. 

Warning: there are duplicate rows in the output now, I will remove them later.

Warning 2: the generations may continue beyond a stop token, degenerating into complete nonsense (code, crazy text, gibberish). A postprocessing script will be added to cut answers of the model when the first stop signal occurs (`<\s>` or `[STOP]`, maybe other tokens). 

The model used is `mistralai/Pixtral-12B-2409` (note: from official huggingface, and not the community edition).

The output format currently in the answer field is not the number of the correct answer but the correct answer as a string. 

**TODO: postprocessing script**