#!/bin/bash
#SBATCH --job-name=qtchlSciQA
#SBATCH --output=job_output_qtchl_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_4
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_qwen_model.py --ngpus 1 --setting QTCHL --resume 3054
