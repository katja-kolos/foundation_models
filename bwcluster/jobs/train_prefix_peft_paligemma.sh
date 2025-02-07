#!/bin/bash
#SBATCH --job-name=PEFTplgG
#SBATCH --output=job_output_pref_paligemma_PEFT_golden_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_4_a100,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_peft.py --num_epochs 20 --model_name "google/paligemma2-3b-pt-224" --exp_setting golden
