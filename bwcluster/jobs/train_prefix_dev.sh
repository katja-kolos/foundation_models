#!/bin/bash
#SBATCH --job-name=PrefDevA100
#SBATCH --output=job_output_pref_dev_a100_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_experiment.py --batch_size 8 --num_epochs 1
