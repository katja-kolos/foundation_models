#!/bin/bash
#SBATCH --job-name=Pref2Plgm
#SBATCH --output=job_output_pref_paligemma_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_8,gpu_4,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python perform_prefix_tuning_experiment.py --ngpus 1 --model_name "google/paligemma2-3b-pt-224" --batch_size 2
