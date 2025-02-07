#!/bin/bash
#SBATCH --job-name=RAWqtchSciQA
#SBATCH --output=job_output_qtch_raw_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 1 --setting QTCH --resume 2315 --split validation
