#!/bin/bash
#SBATCH --job-name=RAWqtchlsSciQA
#SBATCH --output=job_output_qtchls_raw_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_4,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 1 --setting QTCHLS --resume 3194 --split validation
