#!/bin/bash
#z_bins=(0.0 1.0 2.0 3.0 4.0)
#z_bins=(2.9 3.5 5.0)
#z_bins=(0.8 1.2)
z_bins=(0.8 2.1)
G_max="20.5"
save_tag=""

for ((bb=0; bb<$((${#z_bins[@]}-1)); bb++)); do
    fn_gcat_zbin="../data/quaia_G${G_max}_zmin${z_bins[bb]}zmax${z_bins[bb+1]}${save_tag}.fits"
    fn_selfunc="../data/maps/selection_function_NSIDE64_G${G_max}_zmin${z_bins[bb]}zmax${z_bins[bb+1]}${save_tag}.fits"
    
    cat > slurm_jobs/slurm_jobs_zbins/sel_func_zbin${bb}.sh << EOF
#!/bin/bash
#SBATCH --job-name=sel_func_G${G_max}_zmin${z_bins[bb]}zmax${z_bins[bb+1]}${save_tag}
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
##SBATCH --mem=175GB
#SBATCH --mem=360GB
#SBATCH --time=72:00:00

echo "Starting batch job"
cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity exec --overlay \$overlay_ext3:ro /scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash -c "source /ext3/env.sh; cd /home/ksf293/gaia-quasars-lss/code; conda activate gaiaenv; python selection_function_map.py $fn_gcat_zbin $fn_selfunc -p ../data/quaia_G${G_max}.fits"
EOF
    
    sbatch slurm_jobs/slurm_jobs_zbins/sel_func_zbin${bb}.sh
done
