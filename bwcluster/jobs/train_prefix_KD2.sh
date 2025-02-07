#!/bin/bash
#SBATCH --job-name=Pref2KD
#SBATCH --output=job_output_pref_kd2_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=gpu_4,gpu_8,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_experiment.py --exp_setting teacher --batch_size 2
