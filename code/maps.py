import numpy as np
import os
import pandas as pd

import healpy as hp

from astropy.coordinates import SkyCoord
from astropy import units as u

import utils


def main():
    NSIDE = 256
    fn_starmap = f'../data/maps/map_stars_NSIDE{NSIDE}.npy'
    map_stars = maps.get_star_map(NSIDE=NSIDE, fn_map=fn_starmap)



def get_map(NSIDE, ra, dec, quantity=None, func_name='count',
            null_val=0):
    assert func_name in ['count', 'mean'], f"Function {func_name} not recognized!"

    NPIX = hp.nside2npix(NSIDE)
    pixel_indices = hp.ang2pix(NSIDE, ra.value, dec.value, lonlat=True)

    # via https://stackoverflow.com/a/23914036
    # and https://stackoverflow.com/a/58600295
    counts_by_pixel = np.bincount(pixel_indices, minlength=NPIX)
    if func_name=='count':
        map = counts_by_pixel
    elif func_name=='mean':
        # sum of quantity over pixel
        map = np.zeros(NPIX)
        np.add.at(map, pixel_indices, quantity)
        #map = np.bincount(pixel_indices, weights=quantity, minlength=NPIX)
        # divide by counts to get mean
        map /= counts_by_pixel 

    # If we want to put in values other than 0 where no data,
    # need to do like this bc bincount just does 0
    if null_val != 0:
        pixel_arr = np.arange(NPIX)
        pixels_nodata = list(set(pixel_arr) - set(pixel_indices))
        map[pixels_nodata] = null_val
    
    return map, pixel_indices

    
def get_star_map(NSIDE=None, fn_map=None, fn_stars='../data/stars_gaia_G18.5-20.0_rand3e7.fits.gz'):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"Star map already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None, f"{fn_map} doesn't exist; must pass NSIDE to generate!"
    print(f"Generating new star map ({fn_map})")
    tab_stars = load_table(fn_stars)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_stars, _ = get_map(NSIDE, tab_stars['ra'], tab_stars['dec'], 
                                   func_name='count', null_val=0)
    if fn_map is not None:
        np.save(fn_map, map_stars)
        print(f"Saved star map to {fn_map}")
    return map_stars

### Completeness model map functions

def get_completeness(ra, dec, gmag, fn_params='../data/completeness_model_params.dat'):
    """ Get the completeness for a a given value or list 
    of ra(s), dec(s), G-band magnitude(s)

    Parameters
    ----------
    ra : nd.array
        RA at which to evaluate the completeness model
    dec : nd.array
        dec at which to evaluate the completeness model
    gmag : nd.array
        G-band magnitude at which to evaluate the completeness model
    fn_params : str (optional)
        file path to model parameters to pass to selectionFunction function
    Returns
    -------
    nd.array
        Evaluation of the completeness model, in the range [0,1].
    """

    model_params = np.loadtxt(fn_params)

    m10 = get_m10(ra, dec)
    return selectionFunction(gmag, m10, model_params)


def get_m10(ra, dec, fn_comp='../data/completeness_allsky_m10_hpx7.h5'):
    dfm10 = pd.read_hdf(fn_comp, "data")
    pixel_indices = hp.ang2pix(128, ra, dec, lonlat=True, nest=True)
    m10 = dfm10[pixel_indices]
    return m10


def sigmoid(G, G0, invslope, shape):
    """ Generalized sigmoid function 
    
    Parameters
    ----------
    G: nd.array
        where to evaluate the function
    G0: float
        inflection point
    invslope: float
        steepness of the linear part. Shallower for larger values
    shape: float
        if shape=1, model is the classical logistic function, 
        shape converges to zero, then the model is a Gompertz function.

    Returns
    -------
    f(G) evaluation of the model. 
    """
    delta = G - G0
    return 1 - (0.5 * (np.tanh(delta/invslope) + 1)) ** shape


def selectionFunction(G,m10,model_params):
    """ Predicts the completeness at magnitude G, given a value of M_10 read from a precomputed map. 
    
    Parameters
    ----------
    G:   nd.array
            where to evaluate the function
    m10: float
            the value of M_10 in a given region
    model_params: nd.array
            the stored parameters of the trained model

    Returns
    -------
    sf(G) between 0 and 1. 
    """
    a,b,c,d,e,f,x,y,z,lim,sigma = model_params
    
    def sigmoid_params_belowlim(m10_vals):
        pG0 = a*m10_vals+b
        pInvslope = x*m10_vals+y
        pShape = d*m10_vals+e
        return pG0, pInvslope, pShape

    def sigmoid_params_abovelim(m10_vals):
        pG0 = c*m10_vals + (a-c)*lim + b
        pInvslope = z*m10_vals + (x-z)*lim + y
        pShape = f*m10_vals + (d-f)*lim + e
        return pG0, pInvslope, pShape
    
    N_m10 = len(m10)
    predictedG0 = np.empty(N_m10)
    predictedInvslope = np.empty(N_m10)
    predictedShape = np.empty(N_m10)

    idx_above_lim = (m10 > lim)
    idx_below_lim = np.invert(idx_above_lim)
    predictedG0[idx_below_lim], predictedInvslope[idx_below_lim], \
            predictedShape[idx_below_lim] = sigmoid_params_belowlim(m10[idx_below_lim])
    predictedG0[idx_above_lim], predictedInvslope[idx_above_lim], \
            predictedShape[idx_above_lim] = sigmoid_params_abovelim(m10[idx_above_lim])
        
    return sigmoid(G, predictedG0, predictedInvslope, predictedShape)


def get_m10_map(NSIDE, fn_map=None):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"M10 map already exists, loading from {fn_map}")
        return np.load(fn_map)
    print(f"Generating new m10 map ({fn_map})")
    # fix this NSIDE to make dust map determinisitic 
    NSIDE_high = 2048
    NPIX_high = hp.nside2npix(NSIDE_high)
    # get the positions and Av values at the center of a high-NSIDE map
    print("NPIX for dust map sampling:", NPIX_high)
    ra_high, dec_high = hp.pix2ang(NSIDE_high, np.arange(NPIX_high), lonlat=True)
    m10_high = get_m10(ra_high, dec_high)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_m10mean, _ = get_map(NSIDE, ra_high*u.deg, dec_high*u.deg, quantity=m10_high, 
                                         func_name='mean', null_val=np.nan)
    if fn_map is not None:
        np.save(fn_map, map_m10mean)
        print(f"Saved dust map to {fn_map}")
    return map_m10mean

### Dust map functions

# only do this the first time!
fetch_map = False
def fetch_dustmap(map_name='sfd', data_dir='../data/dustmaps'):
    map_dict = {'sfd': dustmaps.sfd}
    if map_name not in map_dict:
        raise ValueError(f"Map name {map_name} not recognized!")
    from dustmaps.config import config
    config['data_dir'] = data_dir

    import dustmaps
    import dustmaps.sfd
    map_dict[map_name].fetch()



def get_dust_map(NSIDE=None, R=None, fn_map=None):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"Dustmap already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None and R is not None, f"{fn_map} doesn't exist; must pass NSIDE and R to generate!"
    print(f"Generating new dust map ({fn_map})")
    # fix this NSIDE to make dust map determinisitic 
    NSIDE_high = 2048
    NPIX_high = hp.nside2npix(NSIDE_high)
    # get the positions and Av values at the center of a high-NSIDE map
    print("NPIX for dust map sampling:", NPIX_high)
    ra_high, dec_high = hp.pix2ang(NSIDE_high, np.arange(NPIX_high), lonlat=True)
    av_high = utils.get_extinction(ra_high*u.deg, dec_high*u.deg, R=R)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_avmean, _ = get_map(NSIDE, ra_high, dec_high, quantity=av_high, 
                                         func_name='mean', null_val=np.nan)
    if fn_map is not None:
        np.save(fn_map, map_avmean)
        print(f"Saved dust map to {fn_map}")
    return map_avmean


if __name__=='__main__':
    main()