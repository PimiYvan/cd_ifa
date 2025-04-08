#!/bin/bash
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# The following three commands allow us to take advantage of whole-node scheduling
# ---------------------------------------------------------------------
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --account=def-menna
# ---------------------------------------------------------------------
# Wall time
# ---------------------------------------------------------------------
#SBATCH --time=00:10:00
# ---------------------------------------------------------------------
# Job output
# ---------------------------------------------------------------------
JOB_NAME="nvidia_smi"
OUTPUT_DIR="output"
OUTPUT="/$OUTPUT_DIR/$JOB_NAME"
mkdir -p $OUTPUT
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT/job_id.txt
echo ""
echo "Saving output log in $OUTPUT/job_id_%j.txt"
echo ""
# ---------------------------------------------------------------------
# Emails me when job starts, ends or fails
# ---------------------------------------------------------------------
#SBATCH --mail-user=ormorteey@gmail.com
#SBATCH --mail-type=ALL
# ---------------------------------------------------------------------
# Echo job details
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo ""
echo "Starting $JOB_NAME job at: $(date)"
echo "Downloading dataset"

# echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
# echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
# echo ""
# ---------------------------------------------------------------------
# Load CCDB modules simulation / Python script
# ---------------------------------------------------------------------
module load python/3.9
# module load scipy-stack
# ---------------------------------------------------------------------
# Set up virtual environment
# ---------------------------------------------------------------------
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# ---------------------------------------------------------------------
# Install python packages / wheels
# ---------------------------------------------------------------------
pip install --no-index torch #torchvision 
pip install --no-index torchvision #torchvision  
pip install --no-index tensorboardX 

# Finish donwloading the dataset 

# ---------------------------------------------------------------------
# Run simulation / Python script
# ---------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
cd cd_fss 
python test.py --backbone resnet50 --benchmark fss --nshot 1 --load  "logs/my-logs.log/best_model.pt"
# python test.py
nvidia-smi