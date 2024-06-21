import argparse
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


"""
Generate a new random catalog the following command on the commandline: 
python generate_random.py <fn_catalog> <fn_selfunc>

<fn_selfunc>: the filepath to the selection function map used to generate 
the random catalog. must be a healpy map saved in fits format with values 
from 0 to 1 (e.g. as generated by selection_function_map.py)

<NSIDE_map>: the NSIDE of the selection function map passed in fn_selfunc.

<fn_rand>: the filepath for the output random catalog.

<fn_catalog> (optional): the filepath to a catalog, used only to calculate the 
target number of random sources (N_catalog*fac_rand). must be in fits format.
If this isn't passed, the default N_rand_target will be used.

For more control, you can edit the settings in the "run" function call
within parse_args.
These settings are: the factor relating the target number of 
random sources to the sources in the given catalog.
"""

def parse_args():
    parser=argparse.ArgumentParser(description="make selection function map for input catalog")
    parser.add_argument("fn_selfunc", type=str)
    parser.add_argument("NSIDE_map", type=int)
    parser.add_argument("fn_rand", type=str)
    parser.add_argument("fn_catalog", type=str, nargs='?', default=None)
    args=parser.parse_args()

    print(f"Generating random with fn_selfunc={args.fn_selfunc}, NSIDE_map={args.NSIDE_map}, " \
         f"fn_rand={args.fn_rand}, fn_catalog={args.fn_catalog}")
    run(args.fn_selfunc, args.NSIDE_map, args.fn_rand, fn_catalog=args.fn_catalog)


def main():

    G_max = 20.5
    #tag_cat = '_qeboss'
    tag_cat = ''
    tag_sel = ''

    fac_rand = 1
    NSIDE_map = 64

    # File names (fn_selfunc is selection function map)
    #fn_selfunc = f'../data/maps/selection_function_NSIDE{NSIDE_map}_G{G_max}{tag_cat}{tag_sel}.fits'
    fn_gaia = f'../data/quaia_G{G_max}{tag_cat}.fits'
    #fn_rand = f'../data/randoms/random_G{G_max}{tag_cat}{tag_sel}_{fac_rand}x_retry.fits'

    fn_selfunc = f'../data/maps/selfunc_test.fits'
    fn_rand = f'../data/randoms/random_test.fits'
    overwrite = True

    run(fn_selfunc, NSIDE_map, fn_rand, fn_catalog=fn_gaia,
        fac_rand=fac_rand, overwrite=overwrite)



def run(fn_selfunc, NSIDE_map, fn_rand, fn_catalog=None, fac_rand=1,
        N_rand_target=1000000, overwrite=False):

    rng = default_rng(seed=42)

    # if a catalog is passed, use it and fac_rand to compute N_rand_target; else,
    # use default N_rand_target value
    if fn_catalog is not None:
    # Load data; only used to figure out desired number
        print("Loading data")
        tab_catalog = utils.load_table(fn_catalog)
        N_data = len(tab_catalog)
        print(f"Number of data sources: {N_data}")
        N_rand_target = int(fac_rand*N_data)

    # first get estimate of reduction factor by starting with some number (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 1000000
    ra_rand_try, dec_rand_try = utils.random_ra_dec_on_sphere(rng, N_rand_try)
    ra_rand, _ = subsample_by_probmap(NSIDE_map, rng, ra_rand_try, dec_rand_try, fn_selfunc)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, using estimated reduction factor to get correct number
    N_rand_init = int(N_rand_target/reduction_factor_estimate)
    ra_rand_init, dec_rand_init = utils.random_ra_dec_on_sphere(rng, N_rand_init)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand = subsample_by_probmap(NSIDE_map, rng, ra_rand_init, dec_rand_init, fn_selfunc)

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


def subsample_by_probmap(NSIDE_map, rng, ra, dec, fn_selfunc,
                         normalize=True):
    map_p = hp.read_map(fn_selfunc)
    print('Sel func min and max:', np.min(map_p), np.max(map_p))
    _, pixel_indices_rand = maps.get_map(NSIDE_map, ra, dec)
    p_rand = map_p[pixel_indices_rand]
    p_rand /= np.max(p_rand)
    #assert np.all(p_rand>=0) and np.all(p_rand<=1), "Bad probability vals!" 
    idx_keep = indices_for_downsample(rng, p_rand)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for probability map")
    return ra[idx_keep], dec[idx_keep]


if __name__=='__main__':
    #main()
    parse_args()
