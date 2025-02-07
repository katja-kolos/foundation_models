#!/bin/bash
#SBATCH --job-name=LoRATuneExp
#SBATCH --output=job_output_lora_%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_4,gpu_4_h100,gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_lora_experiment.py
