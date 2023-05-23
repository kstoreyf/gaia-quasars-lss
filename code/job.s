#!/bin/bash
##SBATCH --job-name=make_data_tables
##SBATCH --job-name=spz_kNN
##SBATCH --job-name=gen_rand_prob_G20.5_10x
##SBATCH --job-name=lensing_qso_cross_G20_NSIDE256_ratio
##SBATCH --job-name=xi_G20_bw4_jack12
#SBATCH --job-name=make_table_G20.6
##SBATCH --job-name=decontaminate_mag0.1_lm5
##SBATCH --job-name=sel_func_G20.5_NSIDE64_mem180_runlong
##SBATCH --job-name=sel_func_G20.0_NSIDE64_fixzeros_mem350_cpu24_hodlr
##SBATCH --job-name=animate_gaia_G20.4_Nall_mp4
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5GB
#SBATCH --time=1:00:00

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
#python animate.py;
#python specphotoz.py;
python make_data_tables.py;
#python lensing_qso_cross.py;
#python correlations.py;
#python decontaminate.py;
#python selection_function_map.py;
#python generate_random.py;
"



