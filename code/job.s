#!/bin/bash
##SBATCH --job-name=make_data_tables
##SBATCH --job-name=spz_kNN
##SBATCH --job-name=gen_rand_prob_G20.5_10x
##SBATCH --job-name=lensing_qso_cross_G20_NSIDE256_ratio
##SBATCH --job-name=xi_G20_bw4_jack12
##SBATCH --job-name=make_table_G20.6
##SBATCH --job-name=decontaminate_mag0.1_lm5_postpm_qeboss
##SBATCH --job-name=sel_func_G20.5_zsplit3bin2
#SBATCH --job-name=sel_func_G20.0_unwise_mem360
##SBATCH --job-name=sel_func_G20.0_NSIDE64_fixzeros_mem350_cpu24_hodlr
##SBATCH --job-name=animate_gcathi_sdss_cbar_setazim
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360GB
#SBATCH --time=24:00:00

# selection function: need mem 175GB. set cpus-per-task=48
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
#python make_data_tables.py;
#python lensing_qso_cross.py;
#python correlations.py;
#python decontaminate.py;
#python selection_function_map.py ../data/quaia_G20.5_zsplit2bin0.fits ../data/maps/selection_function_NSIDE64_G20.5_zsplit2bin0.fits;
#python selection_function_map.py ../data/quaia_G20.0.fits ../data/maps/selection_function_NSIDE64_G20.0.fits;
#python generate_random.py;
python selection_function_map.py
"



