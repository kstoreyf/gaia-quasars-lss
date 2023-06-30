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
import maps


def main():

    fac_rand = 10
    NSIDE_map = 64

    G_max = 20.5
    #tag_cat = '_qeboss'
    tag_cat = ''
    tag_sel = ''

    # File names (fn_map is selection function map usually)
    fn_map = f'../data/maps/selection_function_NSIDE{NSIDE_map}_G{G_max}{tag_cat}{tag_sel}.fits'
    fn_gaia = f'../data/quaia_G{G_max}{tag_cat}.fits'
    fn_rand = f'../data/randoms/random_G{G_max}{tag_cat}{tag_sel}_{fac_rand}x.fits'
    overwrite = True

    rng = default_rng(seed=42)

    # Load data; only used to figure out desired number
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    N_data = len(tab_gaia)
    ra_data, dec_data = tab_gaia['ra'], tab_gaia['dec']
    print(f"Number of data sources: {N_data}")

    # first get estimate of reduction factor by starting with some number (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 1000000
    ra_rand_try, dec_rand_try = utils.random_ra_dec_on_sphere(rng, N_rand_try)
    ra_rand, _ = subsample_by_probmap(NSIDE_map, rng, ra_rand_try, dec_rand_try, fn_map)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, using estimated reduction factor to get correct number
    N_rand_init = int((fac_rand*N_data)/reduction_factor_estimate)
    ra_rand_init, dec_rand_init = utils.random_ra_dec_on_sphere(rng, N_rand_init)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand = subsample_by_probmap(NSIDE_map, rng, ra_rand_init, dec_rand_init, fn_map)

    print(f"Number of final random sources: {len(ra_rand)}")

    # Save! 
    ebv_rand = utils.get_ebv(ra_rand, dec_rand)
    result = [ra_rand, dec_rand, ebv_rand]
    col_names = ['ra', 'dec', 'ebv']
    utils.write_table(fn_rand, result, col_names, overwrite=overwrite)
    print(f"Wrote random to {fn_rand}!")


def indices_for_downsample(rng, probability_accept):
    # if probability is greater than random, keep
    random_vals_rand = rng.random(size=len(probability_accept))
    idx_keep = (probability_accept >= random_vals_rand)
    return idx_keep


def subsample_by_probmap(NSIDE_map, rng, ra, dec, fn_map):
    map_p = hp.read_map(fn_map)
    _, pixel_indices_rand = maps.get_map(NSIDE_map, ra, dec)
    p_rand = map_p[pixel_indices_rand]
    assert np.all(p_rand>=0) and np.all(p_rand<=1), "Bad probability vals!" 
    idx_keep = indices_for_downsample(rng, p_rand)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for probability map")
    return ra[idx_keep], dec[idx_keep]


if __name__=='__main__':
    main()
