#!/bin/bash
#SBATCH --job-name=RAWqtchlSciQAtest
#SBATCH --output=job_output_qtchl_raw_test_%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4,gpu_4,gpu_4_h100,gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 2 --setting QTCHL --resume 2885 --split test
