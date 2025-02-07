#!/bin/bash
#SBATCH --job-name=RAWqtchlsSciQAtest
#SBATCH --output=job_output_qtchls_test_raw_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100,dev_gpu_4
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_model.py --ngpus 1 --setting QTCHLS --resume 3065 --split test
