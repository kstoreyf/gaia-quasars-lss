#!/bin/bash
##SBATCH --job-name=prob_G19.9
##SBATCH --job-name=spz_ANN_G20.5_lr0.005
##SBATCH --job-name=gen_rand_prob_G20.5_10x
##SBATCH --job-name=lensing_qso_cross_G20_NSIDE256_ratio
##SBATCH --job-name=xi_G20_bw4_jack12
##SBATCH --job-name=make_table_gaia_super    
#SBATCH --job-name=decontaminate_mag0.1_lm3
##SBATCH --job-name=sel_func_G20.5_NSIDE64
##SBATCH --job-name=animate_gaia_G20.4_Nall_mp4
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

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
#python specphotoz.py;
#python make_data_tables.py;
#python generate_random.py;
#python lensing_qso_cross.py;
#python correlations.py;
#python selection_function_map.py;
python decontaminate.py;
"



