import numpy as np

import astropy
import healpy as hp
from astropy.table import Table
from scipy.optimize import curve_fit

import utils


def main():

    # save name
    fac_rand = 10
    dust = True
    fn_rand = f'../data/randoms/random_dust_{fac_rand}x.fits'
    overwrite = True

    # about right to get 15 qso per pixel
    NSIDE = 64

    # Load and set up data
    print("Loading data")
    fn_gaia = '../data/gaia_wise_panstarrs_tmass.fits.gz'
    tab_gaia = utils.load_table(fn_gaia)

    # Get only our clean sample (TODO: save this more nicely)
    utils.add_spzs(tab_gaia)
    idx_withspz = np.isfinite(tab_gaia['redshift_spz'])
    tab_gaia_withspz = tab_gaia[idx_withspz]
    N_data = len(tab_gaia_withspz)
    ra_data, dec_data = tab_gaia_withspz['ra'], tab_gaia_withspz['dec']
    print(f"Number of data sources: {N_data}")

    # first get estimate of reduction factor (TODO: better way to do this?)
    print("Estimating reduction factor")
    N_rand_try = 10000
    ra_rand, _ = generate_and_subsample(NSIDE, N_rand_try, ra_data, dec_data, dust=dust)
    reduction_factor_estimate = len(ra_rand)/N_rand_try

    # Generate actual random, dividing by reduction factor
    N_rand_init = int((fac_rand*N_data)/reduction_factor_estimate)
    print(f"Generating random with {fac_rand} times N_data")
    ra_rand, dec_rand = generate_and_subsample(NSIDE, N_rand_init, ra_data, dec_data,
                                               dust=dust)

    # Save! 
    result = [ra_rand, dec_rand]
    col_names = ['ra', 'dec']
    utils.write_table(fn_rand, result, col_names, overwrite=overwrite)
    print(f"Wrote random to {fn_rand}!")


def generate_and_subsample(NSIDE, N_rand_init, ra_data, dec_data, dust=False):
    ra_rand, dec_rand = random_ra_dec_on_sphere(N_rand_init)
    if dust:
        ra_rand, dec_rand = subsample_by_dust(NSIDE, ra_rand, dec_rand, ra_data, dec_data)
    return ra_rand, dec_rand



def subsample_by_dust(NSIDE, ra_rand, dec_rand, ra_data, dec_data, R=3.1):

    # TODO: this needs to be A_G i think, not A_v, but need to figure out that value!

    # generate random just to compute the mean Av in each pixel;
    # think that generally this should be a different random than the one we're making
    ra_avrand, dec_avrand = random_ra_dec_on_sphere(10*len(ra_data))
    av_avrand = utils.get_extinction(ra_avrand, dec_avrand, R=R)
    map_avmean_avrand, _ = utils.get_map(NSIDE, ra_avrand, dec_avrand, quantity=av_avrand, 
                                         func_name='mean', null_val=np.nan)
   
    map_nqso_data, _ = utils.get_map(NSIDE, ra_data, dec_data)
    p_Av = fit_Av(NSIDE, map_avmean_avrand, map_nqso_data, av0_max=0.05)

    # should i be getting these probabilities from exact spot on map,
    # or averaged for some reason?
    # now use the actual random points and use the function to get downsampling val
    av_rand = utils.get_extinction(ra_rand, dec_rand, R=R)
    p_accept = p_Av(av_rand)
    # if probability is greater than random, keep
    random_vals_rand = np.random.rand(len(ra_rand))
    idx_keep = p_accept >= random_vals_rand
    return ra_rand[idx_keep], dec_rand[idx_keep]


def random_ra_dec_on_sphere(N_sphere):
    us = np.random.uniform(size=N_sphere)
    vs = np.random.uniform(size=N_sphere)
    theta_sphere = 2 * np.pi * us
    phi_sphere = np.arccos(2*vs-1)
    
    ra_sphere = theta_sphere * 180/np.pi #+ 180
    dec_sphere = phi_sphere * 180/np.pi - 90
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