#!/bin/bash
#SBATCH --job-name=sel_func_G20.5_zsplit3CIB
#SBATCH --output=logs/%x_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=340GB
##SBATCH --cpus-per-task=1
##SBATCH --mem=1GB
#SBATCH --time=14:00:00
#SBATCH --array=0-2

echo "'My SLURM_ARRAY_TASK_ID:' $SLURM_ARRAY_TASK_ID"
echo "'Running selfunc for' ../data/quaia_G20.5_zsplit3bin${SLURM_ARRAY_TASK_ID}CIB.fits"
echo "'Saving selfunc to' ../data/maps/selection_function_NSIDE64_G20.5_zsplit3bin${SLURM_ARRAY_TASK_ID}CIB.fits"


# selection function: set cpus-per-task=48
# 4 templates: need mem 175GB. 6: meem 340
echo "Starting batch job"
cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
/bin/bash; \
cd /home/ksf293/gaia-quasars-lss/code; \
conda activate gaiaenv; \
python selection_function_map.py ../data/quaia_G20.5_zsplit3bin${SLURM_ARRAY_TASK_ID}CIB.fits ../data/maps/selection_function_NSIDE64_G20.5_zsplit3bin${SLURM_ARRAY_TASK_ID}CIB.fits;
"



