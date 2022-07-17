from getopt import gnu_getopt
import numpy as np
import pandas as pd
from numpy.random import default_rng

import astropy
import healpy as hp
from astropy.table import Table
from astropy import units as u
from scipy.optimize import curve_fit

import utils
import masks


def main():

    # save name
    fac_rand = 10
    dust = True
    completeness = True
    mask_plane = True
    tag_rand = ''
    if dust:
        tag_rand += '_dust'
    if completeness:
        tag_rand += '_completeness'
    if mask_plane:
        tag_rand += '_maskplane'
    fn_rand = f'../data/randoms/random{tag_rand}_{fac_rand}x.fits'
    overwrite = True

    # for getting mean Av_rand; about right to get ~15 qso per pixel
    NSIDE = 64
    rng = default_rng(seed=42)

    # Load and set up data
    print("Loading data")
    fn_gaia = '../data/gaia_spz_kNN.fits'
    tab_gaia = utils.load_table(fn_gaia)
    N_data = len(tab_gaia)
    ra_data, dec_data, gmag_data = tab_gaia['ra'], tab_gaia['dec'], tab_gaia['phot_g_mean_mag']
    print(f"Number of data sources: {N_data}")

    # first get estimate of reduction factor (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 10000
    ra_rand, _ = generate_and_subsample(NSIDE, rng, N_rand_try, ra_data, dec_data, dust=dust)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, dividing by reduction factor
    N_rand_init = int((fac_rand*N_data)/reduction_factor_estimate)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand = generate_and_subsample(NSIDE, rng, N_rand_init, ra_data, dec_data,
                                               dust=dust, completeness=completeness,
                                               mask_plane=mask_plane,
                                               gmag_data=gmag_data)
    print(f"Number of final random sources: {len(ra_rand)}")

    # Save! 
    result = [ra_rand, dec_rand]
    col_names = ['ra', 'dec']
    utils.write_table(fn_rand, result, col_names, overwrite=overwrite)
    print(f"Wrote random to {fn_rand}!")


def generate_and_subsample(NSIDE, rng, N_rand_init, ra_data, dec_data, 
                           dust=False, completeness=False, 
                           mask_plane=False, gmag_data=None):
    ra_rand, dec_rand = random_ra_dec_on_sphere(rng, N_rand_init)
    # should i be masking the data too, before using it to compute Av function?
    if mask_plane:
        b_max = 10
        print(f"Masking galactic plane to b={b_max}")
        ra_rand, dec_rand = subsample_by_mask(NSIDE, ra_rand, dec_rand, masks.galactic_plane_mask, [b_max])
    if dust:
        ra_rand, dec_rand = subsample_by_dust(NSIDE, rng, ra_rand, dec_rand, ra_data, dec_data)
    # should this be sequential, or is there a better way?
    if completeness:
        if gmag_data is None:
            raise ValueError("If getting completess, need to pass a G distribution 'gmag_data'!")
        subsample_by_completeness(NSIDE, rng, ra_rand, dec_rand, gmag_data)
    return ra_rand, dec_rand


def subsample_by_mask(NSIDE, ra_rand, dec_rand, mask_func, mask_func_args):
    mask = mask_func(NSIDE, *mask_func_args)
    _, pixel_indices = utils.get_map(NSIDE, ra_rand, dec_rand)
        
    # TODO: better way to do this??
    pixel_arr = np.arange(len(mask))
    pixel_indices_keep = pixel_arr[mask]
    idx_keep = np.in1d(pixel_indices, pixel_indices_keep, invert=True)
    print(f"Masked {np.sum(idx_keep)/len(idx_keep):.3f} of sources")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def indices_for_downsample(rng, probability_accept):
    # if probability is greater than random, keep
    random_vals_rand = rng.random(size=len(probability_accept))
    idx_keep = (probability_accept >= random_vals_rand)
    return idx_keep


def subsample_by_dust(NSIDE, rng, ra_rand, dec_rand, ra_data, dec_data, R=3.1):
    # TODO: this needs to be R_G to get A_G, now it's R_v!
    # generate random just to compute the mean Av in each pixel;
    # think that generally this should be a different random than the one we're making
    ra_avrand, dec_avrand = random_ra_dec_on_sphere(rng, 10*len(ra_data))
    av_avrand = utils.get_extinction(ra_avrand*u.deg, dec_avrand*u.deg, R=R)
    map_avmean_avrand, _ = utils.get_map(NSIDE, ra_avrand, dec_avrand, quantity=av_avrand, 
                                         func_name='mean', null_val=np.nan)
   
    map_nqso_data, _ = utils.get_map(NSIDE, ra_data, dec_data)
    p_Av = fit_Av(NSIDE, map_avmean_avrand, map_nqso_data, av0_max=0.05)

    # should i be getting these probabilities from exact spot on map,
    # or averaged for some reason?
    # now use the actual random points and use the function to get downsampling val
    av_rand = utils.get_extinction(ra_rand*u.deg, dec_rand*u.deg, R=R)
    p_accept = p_Av(av_rand)
    idx_keep = indices_for_downsample(rng, p_accept)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for dust")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def subsample_by_completeness(NSIDE, rng, ra_rand, dec_rand, gmag_data):
    # pull gmags from data distribution (TODO: better way??)
    gmag_rand = rng.choice(gmag_data, size=len(ra_rand), replace=True)
    completeness = utils.get_completeness(ra_rand, dec_rand, gmag_rand)
    idx_keep = indices_for_downsample(rng, completeness)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for completeness")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def random_ra_dec_on_sphere(rng, N_sphere):
    us = rng.random(size=N_sphere)
    vs = rng.random(size=N_sphere)
    theta_sphere = 2 * np.pi * us
    phi_sphere = np.arccos(2*vs-1)
    
    ra_sphere, dec_sphere = utils.spherical_to_radec(theta_sphere, phi_sphere)
    return ra_sphere, dec_sphere


def prob_Av(A_v, norm): 
    return np.exp(-A_v/norm)


# av0_max is what counts as close enough to zero extinction, to get the
# average with low extinction to set to full completeness
def fit_Av(NSIDE, map_av, map_nqso, av0_max=0.05):
    area_per_pixel = hp.pixelfunc.nside2pixarea(NSIDE, degrees=True)
    map_ndens_qso = map_nqso/area_per_pixel
    ndens_qso_at_av0 = np.mean(map_ndens_qso[map_av < av0_max])
    prob_reduced = map_ndens_qso/ndens_qso_at_av0
    return fit_subsample_prob(prob_Av, map_av, prob_reduced, params_initial_guess=[1])


def fit_subsample_prob(function, x, y, params_initial_guess):
    # if resolution is high, might still have nulls; ignore? TODO figure out if ok
    idx_notnan = np.isfinite(x) & np.isfinite(y)
    params_best_fit, _ = curve_fit(function, x[idx_notnan], y[idx_notnan],
                                   p0=params_initial_guess)
    def subsample_func(map):
        return function(map, *params_best_fit)
    return subsample_func


if __name__=='__main__':
    main()