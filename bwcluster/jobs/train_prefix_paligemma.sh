#!/bin/bash
#SBATCH --job-name=PrfPlgPKVG
#SBATCH --output=job_output_pref_paligemma_pkv_golden_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_4_h100,gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_experiment.py --model_name "google/paligemma2-3b-pt-224" --exp_setting golden --batch_size 4 --num_epochs 20
