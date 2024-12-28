#!/bin/bash
#SBATCH --job-name=PixtralSciQA
#SBATCH --output=job_output_%j.log
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_4
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_model.py --ngpus 4 --setting QTCH --resume 320
python inference_model.py --ngpus 4 --setting QTCHLS --resume 207
