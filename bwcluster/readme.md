# What is here

This folder contains scripts and jobs that were run on [BWUniCluster](https://wiki.bwhpc.de/e/BwUniCluster2.0) for: 

- inferencing existing models for benchmarking purposes with `vllm` (e.g. `Pixtral 12B` which requires more than one GPU)
- training adapters for small models, both in *golden* and *teacher* experimental settings
- scoring the dataset with small models + adapaters

Note: in the end, only prefix-tuning experiments were performed on BW GPU Cluster; LoRA adapter was trained and scored separately (see notebooks). 

# How it runs
The scripts, which are now grouped into folders, were initially in a flat structure in one home folder. 
For each model type, the script slightly differs: pre-processing and post-processing is different; some models require login to Hugginface. 

1. A script for model inference / adapter training is a `.py` script which additionally accepts arguments for configuration (e.g. number of GPUs to use, loading part of the model to CPU etc). 
2. This script is called by an `.sh` script, which also specifies job parameters (asks for resources with a corresponding amount of GPUs, asks for a specific type of GPU -- e.g. with `NVIDIA H100 PCIe` we did not face OOM issues which we did with `Tesla V100-SXM2-32GB` for prefix tuning; for benchmarking, any GPU could be used, because parts of the model could now be loaded to CPU, only resulting in slower processing). Those requirements are chosen by a human launching the job based on what resources are available on the cluster at the moment.
3. The job is submitted by a `sbatch` command and executed when the requested resources are available.
See more on SLURM at BWUniCluster [here](https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues).

# Output assets
- For scoring, output `csvs` are created row by row (not in batches), for each of the four experimental settings: _QTCH, QTCHL, QTCHLS, QTCHS_.
- For `PEFT` prefix training, we saved a `json` with errors after each epoch, as well as a `pth` checkpoint. 
- For debugging purposes, the scripts will create logs (starting with `job_output...` prefix). 

# More on Prefix Tuning
We attempted two versions of Prefix Tuning: one of our own implementation (script names with `PKV`) and one based on the `PEFT` library. 
We observed similar trends (e.g. KD performed slightly worse than training on golden data; validation error was +- similar for both, while train error was larger for KD). 
We decided to go with the `PEFT` implementation in the end to have a more consistent comparison with `LoRA`, for which only `PEFT` was tried. 
