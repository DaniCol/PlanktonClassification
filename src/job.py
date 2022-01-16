import os

def makejob(path_to_config,name):
    return f"""#!/bin/bash 

#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err

python3 train.py --path_to_config {path_to_config}
"""

def submit_job(job):
    with open('job.sbatch', 'w') as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
submit_job(makejob("./config.yaml", "SkuSkuSku"))
#submit_job(makejob("./config-2.yaml", "SkuSkuSku2"))
