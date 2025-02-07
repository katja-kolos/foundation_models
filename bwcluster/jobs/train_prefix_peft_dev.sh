#!/bin/bash
#SBATCH --job-name=PrefPEFT
#SBATCH --output=job_output_pref_PEFT_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --partition=dev_gpu_4_a100,gpu_4_a100,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_peft.py --num_epochs 1 --batch_size 8
