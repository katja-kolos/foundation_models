#!/bin/bash
#SBATCH --job-name=qtchPEFTplgKDtest
#SBATCH --output=job_output_qtch_peft_paligemma_teacher_test_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --partition=gpu_4,gpu_8,gpu_4_h100,gpu_4_a100
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_paligemma_pref_model.py --setting QTCH --resume 0 --split test --local_path "google_paligemma2-3b-pt-224_teacher" --checkpoint 57580
