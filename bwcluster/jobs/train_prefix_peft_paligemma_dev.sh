#!/bin/bash
#SBATCH --job-name=PrfPlgPEFT
#SBATCH --output=job_output_pref_PEFT_plg_dev_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_4_a100,gpu_4_h100,dev_gpu_4_a100,dev_gpu_4
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_peft.py --model_name "google/paligemma2-3b-pt-224" --batch_size 2 --num_epochs 2
