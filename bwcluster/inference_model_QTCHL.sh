#!/bin/bash
#SBATCH --job-name=qtchlPixtralSciQA
#SBATCH --output=job_output_qtchl_%j.log
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_qwen_model.py --ngpus 4 --setting QTCHL --resume 0
