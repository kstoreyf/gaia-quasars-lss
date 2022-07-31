#!/bin/bash
##SBATCH --job-name=spz_G20.5
##SBATCH --job-name=gen_rand_prob_G20_1x
#SBATCH --job-name=lensing_qso_cross_G20_NSIDE2048
##SBATCH --job-name=xi_G20_bw4_jack12
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50GB
#SBATCH --time=24:00:00

cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
/bin/bash; \
cd /home/ksf293/gaia-quasars-lss/code; \
conda activate gaiaenv; \
#python animate.py;
#python specphotoz_knn.py;
#python generate_random.py;
python lensing_qso_cross.py;
#python correlations.py;
"



