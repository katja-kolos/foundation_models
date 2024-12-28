#!/bin/bash
#SBATCH --job-name=qtchlsPixtralSciQA
#SBATCH --output=job_output_qtchls_%j.log
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --partition=gpu_4
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_model.py --ngpus 4 --setting QTCHLS --resume 208
