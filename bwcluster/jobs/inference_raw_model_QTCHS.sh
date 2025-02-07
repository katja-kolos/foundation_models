#!/bin/bash
#SBATCH --job-name=RAWqtchsSciQA
#SBATCH --output=job_output_qtchs_raw_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 1 --setting QTCHS --resume 2328 --split validation
