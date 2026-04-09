#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_short            # <-- Change to your partition
#SBATCH --job-name=abstract-classifier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --oversubscribe

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="poly_datasets"             # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs" "$workdir/.cache" "$workdir/tai"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/abstract-classifier.${SLURM_JOB_ID}.out"
err="$workdir/run_outputs/abstract-classifier.${SLURM_JOB_ID}.err"

#############################################
# Environment Setup
#############################################
# This is necessary to ensure the correct modules and environment are loaded.
export MODULEPATH=/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core
# Purge all modules to start with a clean environment, then load the necessary modules for your job.
module purge
# Load the appropriate CUDA and Python modules.
module load CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0

# python3 -m venv "$workdir/.venv_tai"
source "$workdir/.venv_tai/bin/activate"

# pip3 install --upgrade pip --no-cache-dir
# pip3 install torch==2.8.0 --no-cache-dir
# pip3 install torchaudio==2.8.0 --no-cache-dir
# pip3 install torchvision==0.23.0 --no-cache-dir
# pip3 install transformers --no-cache-dir
# pip3 install vllm --no-cache-dir
# pip3 install pandas --no-cache-dir

# Set environment variables for caching and GPU configuration
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

# Always good to log the environment and job details!
echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################

python3 "$workdir/tai/classifier.py" \
    --model_name "Qwen/Qwen3-8B" \
    --dataset_path "$workdir/tai/abstracts.json" \
    --output_dir "$workdir/tai" \
    --output_file "classified_abstracts.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --gpu_memory_utilization 0.9 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
