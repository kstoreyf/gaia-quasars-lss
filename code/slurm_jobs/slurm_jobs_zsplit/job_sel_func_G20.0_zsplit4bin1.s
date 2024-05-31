#!/bin/bash
#SBATCH --job-name=sel_func_G20.0_zsplit4bin1
#SBATCH --output=slurm_jobs/slurm_jobs_zsplit/logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=250GB
#SBATCH --time=48:00:00

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
python selection_function_map.py ../data/quaia_G20.0_zsplit4bin1.fits ../data/maps/selection_function_NSIDE64_G20.0_zsplit4bin1.fits -p ../data/quaia_G20.0.fits;
"
