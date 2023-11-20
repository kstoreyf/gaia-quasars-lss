#!/bin/bash
#SBATCH --job-name=sel_func_G20.5_zsplit2
#SBATCH --output=logs/%x_%a.out
#SBATCH --nodes=1
##SBATCH --cpus-per-task=48
##SBATCH --mem=340GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --time=18:00:00
#SBATCH --array=0-1

# remember to change: --job_name, --array, n_zbins, tag_cat_extra
n_zbins=2
#tag_cat_extra="CIB"
tag_cat_extra=""
echo "'My SLURM_ARRAY_TASK_ID:' $SLURM_ARRAY_TASK_ID"
echo "'Running selfunc for' ../data/quaia_G20.5_zsplit${n_zbins}bin${SLURM_ARRAY_TASK_ID}${tag_cat_extra}.fits"
echo "'Saving selfunc to' ../data/maps/selection_function_NSIDE64_G20.5_zsplit${n_zbins}bin${SLURM_ARRAY_TASK_ID}${tag_cat_extra}.fits"


# selection function: set cpus-per-task=48
# 4 templates: need mem 175GB. 6: mem 340
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
python selection_function_map.py ../data/quaia_G20.5_zsplit${n_zbins}bin${SLURM_ARRAY_TASK_ID}${tag_cat_extra}.fits ../data/maps/selection_function_NSIDE64_G20.5_zsplit${n_zbins}bin${SLURM_ARRAY_TASK_ID}${tag_cat_extra}.fits ../data/quaia_G20.5.fits;
"



