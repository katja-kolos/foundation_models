#!/bin/bash
#SBATCH --job-name=RAWqtchSciQAtest
#SBATCH --output=job_output_qtch_raw_test_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_4_a100,gpu_4,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 1 --setting QTCH --resume 620 --split test
