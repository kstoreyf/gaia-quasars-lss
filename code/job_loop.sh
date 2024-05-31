#!/bin/bash

#declare -a tags=("_G20.0_blim15" "_G20.0_blim30"
#                  "_G20.5_blim15" "_G20.5_blim30")
declare -a tags=("_G20.0_blim30"
                 "_G20.5_blim15" "_G20.5_blim30")
#declare -a tags=("_G20.0_blim15_tinytest")


## now loop through the above array
dir_scripts="slurm_jobs/slurm_jobs_quaia"
for tag in "${tags[@]}"
do
    job_name="sel_func${tag}"
    fn_script="${dir_scripts}/job_${job_name}.s"
    sbatch $fn_script
    echo "Submitted script $fn_script"
done


# G_max="20.5"
# n_zbins=4
# ((zbin_max=n_zbins-1))
# dir_scripts="slurm_jobs/slurm_jobs_zsplit"
# for zbin in $(seq 0 $zbin_max); do
#     job_name="sel_func_G${G_max}_zsplit${n_zbins}bin${zbin}"
#     fn_script="${dir_scripts}/job_${job_name}.s"
#     sbatch $fn_script
#     echo "Submitted script $fn_script"
# done
