import numpy as np
import os
import time
from numpy.random import default_rng

import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy import units as u

from Corrfunc.theory import DD
from Corrfunc.theory import xi
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.utils import convert_rp_pi_counts_to_wp

import utils 
import masks 


def main():
    #single()
    run()


def run():

    #mode = 'jackknife'
    mode = 'single'

    G_max = 20.0
    fn_data = f'../data/gaia_G{G_max}.fits'
    fn_rand = f'../data/randoms/random_stardustm1064_G{G_max}_10x.fits'
    mask_names_gaia = ['mcs', 'dust']
    Av_max = 0.2
    NSIDE_mask = 64

    cosmo = astropy.cosmology.Planck15
    redshift_name_data = 'redshift_spz'
    redshift_name_rand = 'redshift'

    bin_size = 4
    fn_save = f'../data/xis/xi_G{G_max}_bw{bin_size}.npy'

    rmin, rmax = 24, 152
    #rmin, rmax = 24, 176
    #rmin, rmax = 24, 32
    r_edges = np.arange(rmin, rmax+bin_size, bin_size)

    print("Loading data and random")
    rng = default_rng(seed=42)
    tab_data = utils.load_table(fn_data)
    tab_rand = utils.load_table(fn_rand)

    print("Applying masks")
    mask = masks.get_qso_mask(NSIDE_mask, mask_names_gaia, Av_max=Av_max)
    idx_keep_data = masks.subsample_mask_indices(tab_data['ra'], tab_data['dec'], mask)
    tab_data = tab_data[idx_keep_data]
    idx_keep_rand = masks.subsample_mask_indices(tab_rand['ra'], tab_rand['dec'], mask)
    tab_rand = tab_rand[idx_keep_rand]

    print("Add ras, decs")
    coords_rand = SkyCoord(ra=tab_rand['ra'], dec=tab_rand['dec'], frame='icrs')
    tab_rand.add_column(coords_rand.galactic.l, name='l')
    tab_rand.add_column(coords_rand.galactic.b, name='b')

    print("Add x, y, z to data")
    # Can't for random bc need to shuffle redshifts just before 
    add_dist(tab_data, cosmo, redshift_name_data) 
    add_xyz(tab_data) 

    args = [r_edges, rng]
    kwargs = {'fn_save': fn_save}

    print("Computing 3D xi(r) with r_edges =", r_edges)
    if mode=='single':
        compute_cf3d(tab_data, tab_rand, *args, **kwargs)
    elif mode=='jackknife':
        mean, var = utils.jackknife(compute_cf3d, tab_data, tab_rand, *args, **kwargs)



def compute_cf3d(tab_data, tab_rand, r_edges, rng,
                 fn_save=None, jack=None, nthreads=24):
    print("Computing 3D CR")

    print("Shuffling distances")
    # We need to do this distance assignment in the function to make 
    # sure we are only shuffling those in the final catalog, e.g. bc of jackknife.
    # Don't actually shuffle redshifts, work in dists so don't have to reconvert
    idx_shuffle = rng.integers(low=0, high=len(tab_data), size=len(tab_rand))
    tab_rand.add_column(tab_data['dist'][idx_shuffle], name='dist')
    add_xyz(tab_rand) 

    xi = compute_xi(r_edges, tab_data['x'], tab_data['y'], tab_data['z'],
                             tab_rand['x'], tab_rand['y'], tab_rand['z'],
                             nthreads=nthreads)
    if fn_save is not None:
        if jack is not None:
            fn_bits = os.path.splitext(fn_save)
            fn_save = fn_bits[0]+f'_jack{jack:02d}'+fn_bits[1]
        np.save(fn_save, np.array([r_edges, xi]))
        print("Saved r_edges, xi(r) to", fn_save)
    return xi


def add_dist(tab, cosmo, redshift_name):
    dist_Mpc = cosmo.comoving_distance(tab[redshift_name]).to(u.Mpc)
    dist_Mpcperh = utils.Mpc_to_Mpcperh(dist_Mpc, cosmo).value
    tab.add_column(dist_Mpcperh, name='dist')


def add_xyz(tab):
    x, y, z = utils.radec_to_cartesian(tab['dist'], tab['ra'], tab['dec'])
    tab.add_column(x, name='x')
    tab.add_column(y, name='y')
    tab.add_column(z, name='z')
    return x, y, z


def compute_wtheta(theta_edges, ra, dec, ra_rand, dec_rand,
                   return_full_results=False, nthreads=4):
        
    autocorr = 1
    start = time.time()
    DD_theta = DDtheta_mocks(autocorr, nthreads, theta_edges, ra, dec)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    autocorr = 0
    start = time.time()
    DR_theta = DDtheta_mocks(autocorr, nthreads, theta_edges,
                               ra, dec,
                               RA2=ra_rand, DEC2=dec_rand)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    start = time.time()
    autocorr = 1
    RR_theta = DDtheta_mocks(autocorr, nthreads, theta_edges, ra_rand, dec_rand)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    N = len(ra)
    N_rand = len(ra_rand)
    wtheta = convert_3d_counts_to_cf(N, N, N_rand, N_rand,
                                 DD_theta, DR_theta,
                                 DR_theta, RR_theta)
    
    if return_full_results:
        return wtheta, DD_theta, DR_theta, RR_theta
    
    return wtheta


def compute_xi(r_edges, x, y, z, x_rand, y_rand, z_rand,
               return_full_results=False, nthreads=4):
    print("Computing xi(r) with nthreads = ", nthreads)
    start = time.time()
    autocorr=1
    res_dd = DD(autocorr, nthreads, r_edges, x, y, z, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    start = time.time()
    autocorr=0
    res_dr = DD(autocorr, nthreads, r_edges, x, y, z,
                X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')

    start = time.time()
    autocorr=1
    res_rr = DD(autocorr, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    print()
    
    N = len(x)
    N_rand = len(x_rand)
    xi = convert_3d_counts_to_cf(N, N, N_rand, N_rand,
                                     res_dd, res_dr,
                                     res_dr, res_rr)
    
    if return_full_results:
        return xi, res_dd, res_dr, res_rr
    
    return xi


if __name__=='__main__':
    main()
