#!/bin/bash
#SBATCH --job-name=qtchQwenPtTest
#SBATCH --output=job_output_qtch_qwen_test_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_4,gpu_4_h100,gpu_8,gpu_4_h100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_qwen_model.py --ngpus 1 --setting QTCH --resume 0 --split test --model_name "Qwen/Qwen2-VL-2B-Instruct"
