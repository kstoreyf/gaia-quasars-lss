import numpy as np
import os
import pandas as pd

import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.units import Quantity
from astropy.table import Table
from dustmaps.sfd import SFDQuery


def get_fraction_recovered(Y_true, Y_hat, z_err_close):
        return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)


def add_gaia_wise_colors(tab):
    g = tab['phot_g_mean_mag']
    bp = tab['phot_bp_mean_mag']
    rp = tab['phot_rp_mean_mag']
    w1 = tab['w1mpro']
    w2 = tab['w2mpro']

    tab.add_column(g-rp, name='g_rp')
    tab.add_column(bp-g, name='bp_g')
    tab.add_column(bp-rp, name='bp_rp')
    tab.add_column(g-w1, name='g_w1')
    tab.add_column(w1-w2, name='w1_w2')


def add_spzs(tab_gaia, fn_spz='../data/redshifts_spz_kNN.fits'):
    tab_spz = Table.read(fn_spz, format='fits')
    assert np.allclose(tab_gaia['source_id'], tab_spz['source_id']), "Source IDs don't line up! They should by construction"
    tab_gaia.add_column(tab_spz['redshift_spz'], name='redshift_spz')
    tab_gaia.add_column(tab_spz['redshift_sdss'], name='redshift_sdss')
    

def load_table(fn_fits):
    return Table.read(fn_fits, format='fits')


def write_table(fn_table, data_cols, col_names, overwrite=False):
    tab = Table(data_cols, names=col_names)
    tab.write(fn_table, overwrite=overwrite)
    return tab


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


def add_ebv(tab):
    ebv = get_ebv(tab['ra'], tab['dec'])
    tab.add_column(ebv, name='ebv')


def get_ebv(ra, dec):
    sfd = SFDQuery()
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs') 
    ebv = sfd(coords)
    return ebv


def get_extinction(ra, dec, R=3.1):
    ebv = get_ebv(ra, dec)
    return R*ebv


def get_dust_map(NSIDE=None, R=None, fn_dustmap=None):
    if fn_dustmap is not None and os.path.exists(fn_dustmap):
        print(f"Dustmap already exists, loading from {fn_dustmap}")
        return np.load(fn_dustmap)
    assert NSIDE is not None and R is not None, f"{fn_dustmap} doesn't exist; must pass NSIDE and R to generate!"
    print(f"Generating new dust map ({fn_dustmap}")
    # fix this NSIDE to make dust map determinisitic 
    NSIDE_high = 2048
    NPIX_high = hp.nside2npix(NSIDE_high)
    # get the positions and Av values at the center of a high-NSIDE map
    print("NPIX for dust map sampling:", NPIX_high)
    ra_high, dec_high = hp.pix2ang(NSIDE_high, np.arange(NPIX_high), lonlat=True)
    av_high = get_extinction(ra_high*u.deg, dec_high*u.deg, R=R)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_avmean, _ = get_map(NSIDE, ra_high, dec_high, quantity=av_high, 
                                         func_name='mean', null_val=np.nan)
    if fn_dustmap is not None:
        np.save(fn_dustmap, map_avmean)
        print(f"Saved dust map to {fn_dustmap}")
    return map_avmean



# copied from https://stackoverflow.com/questions/49372918/group-numpy-into-multiple-sub-arrays-using-an-array-of-values
def groupby(values, group_indices):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = group_indices.argsort(kind='mergesort')
    values_sorted = values[sidx]
    group_indices_sorted = group_indices[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,group_indices_sorted[1:] != group_indices_sorted[:-1],True])

    # Split input array with those start, stop ones
    values_grouped = [values_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return values_grouped, list(set(group_indices_sorted))


def get_map(NSIDE, ra, dec, quantity=None, func_name='count',
            null_val=0):
    assert func_name in ['count', 'mean'], f"Function {func_name} not recognized!"

    NPIX = hp.nside2npix(NSIDE)
    pixel_indices = hp.ang2pix(NSIDE, ra, dec, lonlat=True)

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


def get_m10_map(NSIDE, fn_m10map=None):
    if fn_m10map is not None and os.path.exists(fn_m10map):
        print(f"Completness map already exists, loading from {fn_m10map}")
        return np.load(fn_m10map)
    print(f"Generating new dust map ({fn_m10map}")
    # fix this NSIDE to make dust map determinisitic 
    NSIDE_high = 2048
    NPIX_high = hp.nside2npix(NSIDE_high)
    # get the positions and Av values at the center of a high-NSIDE map
    print("NPIX for dust map sampling:", NPIX_high)
    ra_high, dec_high = hp.pix2ang(NSIDE_high, np.arange(NPIX_high), lonlat=True)
    m10_high = get_m10(ra_high, dec_high)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_m10mean, _ = get_map(NSIDE, ra_high, dec_high, quantity=m10_high, 
                                         func_name='mean', null_val=np.nan)
    if fn_m10map is not None:
        np.save(fn_m10map, map_m10mean)
        print(f"Saved dust map to {fn_m10map}")
    return map_m10mean

### Coordinates

# following https://mathworld.wolfram.com/SphericalCoordinates.html
def cartesian_to_spherical(x, y, z):
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arctan2(y,x)
    phi     =  np.arccos(z/r)
    return np.array([r, theta, phi])


def spherical_to_cartesian(r, theta, phi):
    x       =  r*np.cos(theta)*np.sin(phi)
    y       =  r*np.sin(theta)*np.sin(phi)
    z       =  r*np.cos(phi)
    return np.array([x, y, z])


def spherical_to_radec(theta, phi):
    ra = theta * 180/np.pi #+ 180
    dec = phi * 180/np.pi - 90  
    return ra, dec 


# TODO: phi here isnt the inverse of above, check!
def radec_to_spherical(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi


def cartesian_to_radec(x, y, z):
    _, theta, phi = cartesian_to_spherical(x, y, z)
    return spherical_to_radec(theta, phi)


def radec_to_cartesian(r, ra, dec):    
    theta, phi = radec_to_spherical(ra, dec)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    return np.array([x, y, z])


def random_ra_dec_on_sphere(rng, N_sphere):
    us = rng.random(size=N_sphere)
    vs = rng.random(size=N_sphere)
    theta_sphere = 2 * np.pi * us
    phi_sphere = np.arccos(2*vs-1)
    
    ra_sphere, dec_sphere = spherical_to_radec(theta_sphere, phi_sphere)
    return ra_sphere, dec_sphere


### Units

def Mpc_to_Mpcperh(distances_Mpc, cosmo):
    return distances_Mpc * cosmo.h


def Mpcperh_to_Mpc(distances_Mpcperh, cosmo):
    return distances_Mpcperh / cosmo.h


# Other datasets

def get_star_map(NSIDE=None, fn_starmap=None, fn_stars='../data/stars_gaia_G18.5-20.0_rand3e7.fits.gz'):
    if fn_starmap is not None and os.path.exists(fn_starmap):
        print(f"Star map already exists, loading from {fn_starmap}")
        return np.load(fn_starmap)
    assert NSIDE is not None, f"{fn_starmap} doesn't exist; must pass NSIDE to generate!"
    print(f"Generating new star map ({fn_starmap})")
    tab_stars = load_table(fn_stars)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_stars, _ = get_map(NSIDE, tab_stars['ra'], tab_stars['dec'], 
                                   func_name='count', null_val=0)
    if fn_starmap is not None:
        np.save(fn_starmap, map_stars)
        print(f"Saved star map to {fn_starmap}")
    return map_stars


def get_m10_map(NSIDE=None, fn_m10map=None, fn_comp='../data/completeness_allsky_m10_hpx7.h5'):
    if fn_m10map is not None and os.path.exists(fn_m10map):
        print(f"Star map already exists, loading from {fn_m10map}")
        return np.load(fn_m10map)
    assert NSIDE is not None, f"{fn_m10map} doesn't exist; must pass NSIDE to generate!"
    print(f"Generating new m10 map ({fn_m10map})")
    
    df_m10 = pd.read_hdf(fn_comp, "data")
    # m10 is in nested order! but we want ring, usually
    map_m10 = hp.ud_grade(np.array(df_m10), NSIDE, order_in='NESTED', order_out='RING')
    if fn_m10map is not None:
        np.save(fn_m10map, map_m10)
        print(f"Saved star map to {fn_m10map}")
    return map_m10