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
    fac_rand = 1
    dust = True
    completeness = False
    mask_plane = False
    mask_mcs = False
    mask_dust = False
    tag_rand = ''

    b_max = 10
    Av_max = 0.2
    gmag_comp = 20.8
    R = 3.1
    NSIDE_dustmap = 8
    #tag_rand = '_seed41'
    if dust:
        tag_rand += f'_dust{NSIDE_dustmap}'
    if completeness:
        tag_rand += f'_completeness{gmag_comp}'
    if mask_plane:
        tag_rand += f'_maskplane{b_max}'
    if mask_mcs:
        tag_rand += '_maskmcs'
    if mask_dust:
        tag_rand += f'_maskdust{Av_max}'
    fn_rand = f'../data/randoms/random{tag_rand}_{fac_rand}x.fits'
    tag_data = '_spz_kNN'
    overwrite = True

    NSIDE_masks = 64
    rng = default_rng(seed=42)
    fn_dustmap = f'../data/dustmap_Avmean_NSIDE{NSIDE_dustmap}.npy'

    # Load and set up data
    print("Loading data")
    fn_gaia = f'../data/gaia{tag_data}.fits'
    tab_gaia = utils.load_table(fn_gaia)
    N_data = len(tab_gaia)
    ra_data, dec_data = tab_gaia['ra'], tab_gaia['dec']
    print(f"Number of data sources: {N_data}")

    # first get estimate of reduction factor (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 10000
    ra_rand, _ = generate_and_subsample(NSIDE_dustmap, rng, N_rand_try, ra_data, dec_data, 
                                        dust=dust, completeness=completeness, 
                                        fn_dustmap=fn_dustmap, gmag_comp=gmag_comp, R=R)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, dividing by reduction factor
    N_rand_init = int((fac_rand*N_data)/reduction_factor_estimate)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand =  generate_and_subsample(NSIDE_dustmap, rng, N_rand_init, ra_data, dec_data, 
                                        dust=dust, completeness=completeness, 
                                        fn_dustmap=fn_dustmap, gmag_comp=gmag_comp, R=R)
    ra_rand, dec_rand = apply_masks(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                                    mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                    mask_dust=mask_dust,
                                    fn_dustmap=fn_dustmap, b_max=b_max,
                                    Av_max=Av_max, R=R)
    print(f"Number of final random sources: {len(ra_rand)}")

    # Save! 
    result = [ra_rand, dec_rand]
    col_names = ['ra', 'dec']
    utils.write_table(fn_rand, result, col_names, overwrite=overwrite)
    print(f"Wrote random to {fn_rand}!")


def generate_and_subsample(NSIDE_dustmap, rng, N_rand_init, ra_data, dec_data, 
                           dust=False, completeness=False, 
                           fn_dustmap=None, gmag_comp=20.8, R=3.1):
    # gmag=20.8 is where completeness is 91.4% on average
    ra_rand, dec_rand = utils.random_ra_dec_on_sphere(rng, N_rand_init)
    # should this be sequential, or is there a better way?
    # should i be masking the data too, before using it to compute Av function?
    if dust:
        ra_rand, dec_rand = subsample_by_dust(NSIDE_dustmap, rng, ra_rand, dec_rand, ra_data, dec_data,
                                              R=R, fn_dustmap=fn_dustmap)
    if completeness:
        ra_rand, dec_rand = subsample_by_completeness(rng, ra_rand, dec_rand, gmag_comp)
    return ra_rand, dec_rand


def apply_masks(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                mask_plane=False, mask_mcs=False, 
                mask_dust=None,
                fn_dustmap=None, b_max=10,
                Av_max=0.2, R=3.1):
    if mask_plane:
        print(f"Masking galactic plane to b={b_max}")
        ra_rand, dec_rand = masks.subsample_by_mask(NSIDE_masks, ra_rand, dec_rand, 
                                                    masks.galactic_plane_mask, [b_max])
    if mask_mcs:
        ra_rand, dec_rand = masks.subsample_by_mask(NSIDE_masks, ra_rand, dec_rand, 
                                                    masks.magellanic_clouds_mask, [])
    if mask_dust:
        ra_rand, dec_rand = masks.subsample_by_mask(NSIDE_dustmap, ra_rand, dec_rand, 
                                                    masks.galactic_dust_mask, 
                                                    [Av_max, R, fn_dustmap])                               
    return ra_rand, dec_rand


def indices_for_downsample(rng, probability_accept):
    # if probability is greater than random, keep
    random_vals_rand = rng.random(size=len(probability_accept))
    idx_keep = (probability_accept >= random_vals_rand)
    return idx_keep


def subsample_by_dust(NSIDE, rng, ra_rand, dec_rand, ra_data, dec_data, R=3.1, fn_dustmap=None):
    # TODO: this needs to be R_G to get A_G, now it's R_v!
    # generate random just to compute the mean Av in each pixel;
    # think that generally this should be a different random than the one we're making
    map_avmean = utils.get_dust_map(NSIDE, R, fn_dustmap=fn_dustmap)
   
    map_nqso_data, _ = utils.get_map(NSIDE, ra_data, dec_data)
    p_Av = fit_Av(NSIDE, map_avmean, map_nqso_data, av0_max=0.05)

    # should i be getting these probabilities from exact spot on map,
    # or averaged for some reason?
    # now use the actual random points and use the function to get downsampling val
    av_rand = utils.get_extinction(ra_rand*u.deg, dec_rand*u.deg, R=R)
    p_accept = p_Av(av_rand)
    idx_keep = indices_for_downsample(rng, p_accept)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for dust")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def subsample_by_completeness(rng, ra_rand, dec_rand, gmag_comp):
    # pull gmags from data distribution (TODO: better way??)
    #gmag_rand = rng.choice(gmag_data, size=len(ra_rand), replace=True)
    completeness = utils.get_completeness(ra_rand, dec_rand, gmag_comp)
    assert np.all(completeness>=0) and np.all(completeness<=1), "Bad completeness vals!" 
    idx_keep = indices_for_downsample(rng, completeness)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for completeness")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def prob_Av(A_v, norm): 
    return np.exp(-A_v/norm)


# av0_max is what counts as close enough to zero extinction, to get the
# average with low extinction to set to full completeness
def fit_Av(NSIDE, map_av, map_nqso, av0_max=0.02):
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