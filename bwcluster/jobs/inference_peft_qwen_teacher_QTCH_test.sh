#!/bin/bash
#SBATCH --job-name=qtchPEFTqwenKD
#SBATCH --output=job_output_qtch_peft_qwen_teacher_test_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=06:30:00
#SBATCH --partition=gpu_4,gpu_8
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_qwen_pref_model.py --ngpus 1 --setting QTCH --resume 0 --split test --local_path "qwen_qwen2-vl-2b-instruct_teacher" --checkpoint 57580
