# **************************************************
# * File Name : make_jobs.py
# * Creation Date : 2024-05-15
# * Created By : kstoreyf
# * Description :
# **************************************************

import pathlib


def main():
    #write_zsplit_loop()
    write_loop()


def write_loop():

    dir_scripts = 'slurm_jobs/slurm_jobs_quaia'    
    pathlib.Path(dir_scripts).mkdir(parents=True, exist_ok=True) 
    dir_logs = f'{dir_scripts}/logs'
    pathlib.Path(dir_logs).mkdir(parents=True, exist_ok=True)

    #tags_catmasked = ['_G20.0_blim15', '_G20.0_blim30',
    #                  '_G20.5_blim15', '_G20.5_blim30']
    tags_catmasked = ['_G20.0_blim15']
    tiny_test = True

    if tiny_test:
        cpus = 4
        mem = '20GB'
        time = '1:00:00'
        tag_tiny_test = '_tinytest'
    else:
        cpus = 48
        mem = '300GB'
        time = '72:00:00'

    for tag_catmasked in tags_catmasked:
        job_name = f'sel_func{tag_catmasked}{tag_tiny_test}'
        fn_script = f'{dir_scripts}/job_{job_name}.s'
        command = f'python selection_function_map.py /scratch/aew492/quasars/catalogs/quaia/masked_quaia{tag_catmasked}.fits ../data/maps/selection_function_NSIDE64{tag_catmasked}{tag_tiny_test}.fits --inputs_are_maps'
        if tiny_test:
            command += ' --tiny_test'
        write_slurm_script(fn_script, dir_logs, job_name, cpus, mem, time, command)
    



def write_zsplit_loop():

    dir_scripts = 'slurm_jobs/slurm_jobs_zsplit'    
    pathlib.Path(dir_scripts).mkdir(parents=True, exist_ok=True) 
    dir_logs = f'{dir_scripts}/logs'
    pathlib.Path(dir_logs).mkdir(parents=True, exist_ok=True)

    G_max = '20.5'
    n_zbins = 4
    
    #cpus = 4
    #mem = '20GB'
    #time = '1:00:00'
    cpus = 48
    mem = '250GB'
    time = '48:00:00'

    for zbin in range(n_zbins):
        job_name = f'sel_func_G{G_max}_zsplit{n_zbins}bin{zbin}'
        fn_script = f'{dir_scripts}/job_{job_name}.s'
        command = f'python selection_function_map.py ../data/quaia_G{G_max}_zsplit{n_zbins}bin{zbin}.fits ../data/maps/selection_function_NSIDE64_G{G_max}_zsplit{n_zbins}bin{zbin}.fits -p ../data/quaia_G{G_max}.fits'
        write_slurm_script(fn_script, dir_logs, job_name, cpus, mem, time, command)
    


def write_slurm_script(fn_script, dir_logs, job_name, cpus, mem, time, command,
                       conda_env='gaiaenv',
                       dir_cd='/home/ksf293/gaia-quasars-lss/code',
                       ):
    # CPU/RAM maxes on greene are 48/180GB, 48/369GB
    text = \
f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={dir_logs}/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}

echo "Starting batch job"
cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \\
exec --overlay $overlay_ext3:ro \\
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \\
-c "source /ext3/env.sh; \\
/bin/bash; \\
cd {dir_cd}; \\
conda activate {conda_env}; \\
{command};
"
'''
    with open(fn_script, 'w') as f:
        f.write(text)
    print(f"Wrote script to {fn_script}")


if __name__=='__main__':
    main()

