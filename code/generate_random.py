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

    # save name
    fac_rand = 10
    dust = False
    completeness = False
    stardens = False
    prob_map = True
    mask_plane = False
    mask_mcs = False
    mask_dust = False
    tag_rand = ''
    tag_prob_map = 'stardustm10mcs'

    G_max = 20.0
    b_max = 10
    Av_max = 0.2
    gmag_comp = 20.8
    R = 3.1
    NSIDE_dustmap = 64
    NSIDE_starmap = 64
    NSIDE_probmap = 64
    #tag_rand = '_seed41'
    if dust:
        tag_rand += f'_dust{NSIDE_dustmap}'
    if completeness:
        tag_rand += f'_completeness{gmag_comp}'
    if stardens:
        tag_rand += f'_stardens{NSIDE_starmap}'
    if prob_map:
        tag_rand += f'_{tag_prob_map}_NSIDE{NSIDE_probmap}'
    if mask_plane:
        tag_rand += f'_maskplane{b_max}'
    if mask_mcs:
        tag_rand += '_maskmcs'
    if mask_dust:
        tag_rand += f'_maskdust{Av_max}'

    tag_cat = '_qeboss'
    #fn_rand = f'../data/randoms/random{tag_rand}_{fac_rand}x.fits'
    fn_rand = f'../data/randoms/random_G{G_max}{tag_cat}_{fac_rand}x.fits'
    overwrite = True
    NSIDE_masks = 64
    rng = default_rng(seed=42)
    fn_dustmap = f'../data/dustmap_Avmean_NSIDE{NSIDE_dustmap}.npy'
    fn_starmap = f'../data/stardensmap_NSIDE{NSIDE_starmap}.npy'
    #fn_probmap = f'../data/maps/map_probability_dust_stars_m10_mcs_NSIDE{NSIDE_probmap}_G{G_max}.fits'
    fn_probmap = f'../data/maps/selection_function_NSIDE{NSIDE_probmap}_G{G_max}{tag_cat}.fits'

    # Load and set up data
    print("Loading data")
    fn_gaia = f'../data/catalog_G{G_max}{tag_cat}.fits'
    tab_gaia = utils.load_table(fn_gaia)
    N_data = len(tab_gaia)
    ra_data, dec_data = tab_gaia['ra'], tab_gaia['dec']
    print(f"Number of data sources: {N_data}")

    # first get estimate of reduction factor (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 10000
    ra_rand, _ = generate_and_subsample(NSIDE_dustmap, NSIDE_starmap, 
                                        NSIDE_probmap, rng, 
                                        N_rand_try, ra_data, dec_data, 
                                        dust=dust, completeness=completeness, stardens=stardens, prob_map=prob_map,
                                        fn_dustmap=fn_dustmap, fn_starmap=fn_starmap, 
                                        fn_probmap=fn_probmap,
                                        gmag_comp=gmag_comp, R=R)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, dividing by reduction factor
    N_rand_init = int((fac_rand*N_data)/reduction_factor_estimate)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand =  generate_and_subsample(NSIDE_dustmap, NSIDE_starmap, 
                                        NSIDE_probmap, rng, 
                                        N_rand_init, ra_data, dec_data, 
                                        dust=dust, completeness=completeness, stardens=stardens, prob_map=prob_map,
                                        fn_dustmap=fn_dustmap,  fn_starmap=fn_starmap, 
                                        fn_probmap=fn_probmap, 
                                        gmag_comp=gmag_comp, R=R)
    idx_keep = get_mask_indices(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                                    mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                    mask_dust=mask_dust,
                                    fn_dustmap=fn_dustmap, b_max=b_max,
                                    Av_max=Av_max, R=R)
    ra_rand, dec_rand = ra_rand[idx_keep], dec_rand[idx_keep]
    print(f"Number of final random sources: {len(ra_rand)}")
    print(np.min(ra_rand), np.max(ra_rand))
    print(np.min(dec_rand), np.max(dec_rand))
    # Save! 
    ebv_rand = utils.get_ebv(ra_rand, dec_rand)
    result = [ra_rand, dec_rand, ebv_rand]
    col_names = ['ra', 'dec', 'ebv']
    utils.write_table(fn_rand, result, col_names, overwrite=overwrite)
    print(f"Wrote random to {fn_rand}!")


def generate_and_subsample(NSIDE_dustmap, NSIDE_starmap, NSIDE_probmap, rng, 
                           N_rand_init, ra_data, dec_data, 
                           dust=False, completeness=False, stardens=False, 
                           prob_map=False,
                           fn_dustmap=None, fn_starmap=None, fn_probmap=None,
                           gmag_comp=20.8, R=3.1):
    # gmag=20.8 is where completeness is 91.4% on average
    ra_rand, dec_rand = utils.random_ra_dec_on_sphere(rng, N_rand_init)
    # should this be sequential, or is there a better way?
    # should i be masking the data too, before using it to compute Av function?
    if dust:
        ra_rand, dec_rand = subsample_by_dust(NSIDE_dustmap, rng, ra_rand, dec_rand, 
                                              ra_data, dec_data, R=R, fn_dustmap=fn_dustmap)
    if completeness:
        ra_rand, dec_rand = subsample_by_completeness(rng, ra_rand, dec_rand, gmag_comp)
    if stardens:
        ra_rand, dec_rand = subsample_by_stardens(NSIDE_starmap, rng, ra_rand, dec_rand, 
                                                  ra_data, dec_data, fn_starmap=fn_starmap)
    if prob_map:
        ra_rand, dec_rand = subsample_by_prob_map(NSIDE_probmap, rng, 
                                                  ra_rand, dec_rand, 
                                                  fn_probmap=fn_probmap)
    return ra_rand, dec_rand


def get_mask_indices(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                mask_plane=False, mask_mcs=False, 
                mask_dust=None,
                fn_dustmap=None, b_max=10,
                Av_max=0.2, R=3.1):
    idx_keep = np.full(len(ra_rand),True)
    if mask_plane:
        print(f"Masking galactic plane to b={b_max}")
        idx_keep_m = masks.subsample_by_mask(NSIDE_masks, ra_rand, dec_rand, 
                                                    masks.galactic_plane_mask, [b_max])
        idx_keep = idx_keep & idx_keep_m
    if mask_mcs:
        idx_keep_m = masks.subsample_by_mask(NSIDE_masks, ra_rand, dec_rand, 
                                                    masks.magellanic_clouds_mask, [])
        idx_keep = idx_keep & idx_keep_m
    if mask_dust:
        idx_keep_m = masks.subsample_by_mask(NSIDE_dustmap, ra_rand, dec_rand, 
                                                    masks.galactic_dust_mask, 
                                                    [Av_max, R, fn_dustmap])  
        idx_keep = idx_keep & idx_keep_m                     
    return idx_keep


def indices_for_downsample(rng, probability_accept):
    # if probability is greater than random, keep
    random_vals_rand = rng.random(size=len(probability_accept))
    idx_keep = (probability_accept >= random_vals_rand)
    return idx_keep


def subsample_by_dust(NSIDE, rng, ra_rand, dec_rand, ra_data, dec_data, R=3.1, fn_dustmap=None):
    # TODO: this needs to be R_G to get A_G, now it's R_v!
    # generate random just to compute the mean Av in each pixel;
    # think that generally this should be a different random than the one we're making
    map_avmean = utils.get_dust_map(NSIDE, R, fn_map=fn_dustmap)
   
    map_nqso_data, _ = maps.get_map(NSIDE, ra_data, dec_data)
    p_Av = fit_reduction_vs_quantity(NSIDE, map_avmean, map_nqso_data, val0_max=0.05)

    # should i be getting these probabilities from exact spot on map,
    # or averaged for some reason?
    # now use the actual random points and use the function to get downsampling val
    av_rand = utils.get_extinction(ra_rand, dec_rand, R=R)
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


def subsample_by_prob_map(NSIDE_probmap, rng, ra_rand, dec_rand, fn_probmap=None):
    # pull gmags from data distribution (TODO: better way??)
    #gmag_rand = rng.choice(gmag_data, size=len(ra_rand), replace=True)
    map_p = hp.read_map(fn_probmap)
    _, pixel_indices_rand = maps.get_map(NSIDE_probmap, ra_rand, dec_rand)
    p_rand = map_p[pixel_indices_rand]
    assert np.all(p_rand>=0) and np.all(p_rand<=1), "Bad probability vals!" 
    idx_keep = indices_for_downsample(rng, p_rand)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for probability map")
    return ra_rand[idx_keep], dec_rand[idx_keep]


def subsample_by_stardens(NSIDE, rng, ra_rand, dec_rand, ra_data, dec_data, fn_starmap=None):
    map_stars = maps.get_star_map(NSIDE, fn_map=fn_starmap)
    area_per_pixel = hp.nside2pixarea(NSIDE, degrees=True)
    map_stardens = map_stars/area_per_pixel

    map_nqso_data, _ = maps.get_map(NSIDE, ra_data, dec_data)
    p_stardens = fit_reduction_vs_quantity(NSIDE, map_stardens, map_nqso_data, val0_max=10)

    # here, subsampling based on the healpix map vals (unlike dust, for which we have a finer map!)    
    _, pixel_indices_rand = maps.get_map(NSIDE, ra_rand, dec_rand)
    stardens_rand = map_stardens[pixel_indices_rand]
    p_accept = p_stardens(stardens_rand)
    idx_keep = indices_for_downsample(rng, p_accept)
    print(f"Subsampling by {np.sum(idx_keep)/len(idx_keep):.3f} for stellar density")
    return ra_rand[idx_keep], dec_rand[idx_keep]



def exponential(x, norm): 
    return np.exp(-x/norm)


# av0_max is what counts as close enough to zero extinction, to get the
# average with low extinction to set to full completeness
def fit_reduction_vs_quantity(NSIDE, map_quantity, map_nqso, val0_max=0.02, params_initial_guess=[1]):
    area_per_pixel = hp.pixelfunc.nside2pixarea(NSIDE, degrees=True)
    map_ndens_qso = map_nqso/area_per_pixel
    ndens_qso_at_val0 = np.mean(map_ndens_qso[map_quantity < val0_max])
    prob_reduced = map_ndens_qso/ndens_qso_at_val0
    subsample_func = fit_subsample_prob(exponential, map_quantity, prob_reduced, params_initial_guess=params_initial_guess)
    return subsample_func


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
