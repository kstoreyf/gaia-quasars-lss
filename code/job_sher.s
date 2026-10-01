#!/bin/bash
# Smoke test for Stanford Sherlock Slurm. Submit from this directory after:
#   mkdir -p logs
#   sbatch job_sher.s
#
# Uses the dev partition (short waits, 2h max wall time). Uncomment --account
# if your lab/PI requires a specific Slurm account (see: sacctmgr show user $USER).

#SBATCH --job-name=sherlock_test
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.out
##SBATCH --account=YOUR_LAB_GROUP

set -euo pipefail

echo "=== Sherlock Slurm smoke test ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "Node: $(hostname)"
echo "Date: $(date -Is)"
echo "User: ${USER:-}"
echo "Submit dir (pwd): $(pwd)"

echo "--- srun check (single task) ---"
srun -n 1 hostname

echo "--- Python (if available) ---"
if command -v python3 >/dev/null 2>&1; then
  python3 -c 'import sys; print("python:", sys.version.split()[0])'
else
  echo "python3 not on PATH (ok for a minimal cluster test)"
fi

echo "=== OK: job finished ==="
