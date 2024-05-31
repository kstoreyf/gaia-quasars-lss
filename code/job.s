#!/bin/bash
##SBATCH --job-name=make_data_tables
##SBATCH --job-name=spz_kNN
##SBATCH --job-name=gen_rand_G20.5
##SBATCH --job-name=make_cats
##SBATCH --job-name=lensing_qso_cross_G20_NSIDE256_ratio
##SBATCH --job-name=xi_G20_bw4_jack12
##SBATCH --job-name=make_table_G20.6
##SBATCH --job-name=decontaminate_mag0.1_lm5_postpm_qeboss
##SBATCH --job-name=sel_func_G20.5_zsplit3bin0CIB_tiny
#SBATCH --job-name=sel_func_G20.0_nomcs
##SBATCH --job-name=sel_func_catwise_pluszodis
##SBATCH --job-name=sel_func_catwise_zodi3.4
##SBATCH --job-name=sel_func_G20.5
##SBATCH --job-name=sel_func_G20.0_NSIDE64_fixzeros_mem350_cpu24_hodlr
##SBATCH --job-name=animate_gcathi_sdss_cbar_setazim
##SBATCH --job-name=animate_gcathi_sdss_image_lowmem
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=250GB
##SBATCH --mem=125GB
##SBATCH --cpus-per-task=8
##SBATCH --mem=40GB
#SBATCH --time=72:00:00
# CPU/RAM maxes on greene are 48/180GB, 48/369GB

G_max="20.0"

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
# dust stars m10 mcs unwise unwisescan mcsunwise zodi1.25 zodi3.4 zodi4.6
python selection_function_map.py ../data/quaia_G20.0_zsplit2bin0.fits ../data/maps/selection_function_NSIDE64_G20.0_zsplit2bin0_tiny.fits -p ../data/quaia_G20.0.fits;
#python selection_function_map.py ../data/quaia_G20.0.fits ../data/maps/selection_function_NSIDE64_G20.0_nomcs.fits -m dust stars m10 unwise unwisescan;
#python selection_function_map.py /scratch/aew492/quasars/catalogs/catwise_agns/masked_catwise_agns_master_blim15.fits ../data/maps/selection_function_NSIDE64_catwise_pluszodis.fits -m dust unwise unwisescan zodi3.4 zodi4.6 --inputs_are_maps;
#python generate_random.py ../data/maps/selection_function_NSIDE64_G${G_max}.fits 64 ../data/randoms/random_G${G_max}_10x.fits ../data/quaia_G${G_max}.fits;
tag_selfunc='_pluszodis'
#python generate_random.py ../data/maps/selection_function_NSIDE64_catwise${tag_selfunc}.fits 64 ../data/randoms/random_catwise${tag_selfunc}_1x.fits /scratch/aew492/quasars/catalogs/catwise_agns/catwise_agns_master_masks_w1_b15.fits;
#python selection_function_map.py /scratch/aew492/quasars/catalogs/catwise_agns/catwise_agns_master_masks_w1_b15.fits ../data/maps/selection_function_NSIDE64_catwise_zodis.fits -m zodi3.4 zodi4.6;
#python selection_function_map.py /scratch/aew492/quasars/catalogs/catwise_agns/catwise_agns_master_masks_w1.fits ../data/maps/selection_function_NSIDE64_catwise_dust.fits -m dust;
#python selection_function_map.py ../data/quaia_G20.5.fits ../data/maps/selection_function_NSIDE64_G20.5.fits;
#python selection_function_map.py ../data/quaia_G20.5_zsplit3bin0CIB.fits ../data/maps/selection_function_NSIDE64_G20.5_zsplit3bin0CIB_tiny.fits ../data/quaia_G20.5.fits;
#python make_catalogs.py;
#python animate.py;
#python specphotoz.py;
#python make_data_tables.py;
#python lensing_qso_cross.py;
#python correlations.py;
#python decontaminate.py;
#python selection_function_map.py ../data/quaia_G20.5_zsplit2bin1.fits ../data/maps/selection_function_NSIDE64_G20.5_zsplit2bin1.fits;
#python selection_function_map.py ../data/quaia_G20.5.fits ../data/maps/selection_function_NSIDE64_G20.5.fits;
#python generate_random.py;
#python selection_function_map.py
"

