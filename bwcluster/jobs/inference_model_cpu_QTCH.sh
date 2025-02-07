#!/bin/bash
#SBATCH --job-name=CPUqtchSciQA
#SBATCH --output=job_output_qtch_cpu_%j.log
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --partition=dev_multiple
export PYTHONPATH=/pfs/data5/home/st/st_us-053000/st_st186032/fmenv/lib/python3.9/site-packages
source ~/fmenv/bin/activate
python inference_qwen_model.py --ngpus=0  --setting QTCH --resume 0
