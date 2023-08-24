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
    map_stars = get_star_map(NSIDE=NSIDE, fn_map=fn_starmap)



def get_map(NSIDE, ra, dec, quantity=None, func_name='count',
            null_val=0):
    assert func_name in ['count', 'mean'], f"Function {func_name} not recognized!"

    NPIX = hp.nside2npix(NSIDE)
    if type(ra) == u.Quantity:
        ra = ra.to('deg')
        ra = ra.value
    if type(dec) == u.Quantity:
        dec = dec.to('deg')
        dec = dec.value
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

    
def get_star_map(NSIDE=None, fn_map=None, fn_stars='../data/stars_gaia_G18.5-20.0_rand3e7.fits.gz',
                reverse=True):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"Star map already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None, f"{fn_map} doesn't exist; must pass NSIDE to generate!"
    print(f"Generating new star map ({fn_map})")
    tab_stars = utils.load_table(fn_stars)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_stars, _ = get_map(NSIDE, tab_stars['ra'], tab_stars['dec'], 
                                   func_name='count', null_val=0)
    if fn_map is not None:
        np.save(fn_map, map_stars)
        print(f"Saved star map to {fn_map}")
    return map_stars


def get_unwise_map(NSIDE=None, fn_map=None, fn_unwise='../data/unwise_rand0.01.fits.gz',
                reverse=True):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"unWISE map already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None, f"{fn_map} doesn't exist; must pass NSIDE to generate!"
    print(f"Generating new unWISE map ({fn_map})")
    tab_unwise = utils.load_table(fn_unwise)
    # Take the average over these points, so for a given NSIDE should get exact same map
    map_unwise, _ = get_map(NSIDE, tab_unwise['RAJ2000'], tab_unwise['DEJ2000'], 
                                   func_name='count', null_val=0)
    if fn_map is not None:
        np.save(fn_map, map_unwise)
        print(f"Saved unWISE map to {fn_map}")
    return map_unwise


def get_mcs_map(NSIDE, fn_map=None, fn_stars='../data/stars_gaia_G18.5-20.0_rand3e7.fits.gz',
                fn_starmap=None):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"MCs map already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None, f"{fn_map} doesn't exist; must pass NSIDE to generate!"
    
    if fn_starmap is None:
        fn_starmap = f'../data/maps/map_stars_NSIDE{NSIDE}.npy'

    from astropy.coordinates import Galactic, ICRS

    fn_stars='../data/stars_gaia_G18.5-20.0_rand3e7.fits.gz'
    tab_stars = utils.load_table(fn_stars)

    cat = SkyCoord(tab_stars['ra'], tab_stars['dec'], frame='icrs')
    cat_galactic = cat.transform_to(Galactic())
    cat_galactic_reversed = SkyCoord(360*u.deg-cat_galactic.l, cat_galactic.b, frame='galactic')
    cat_reversed = cat_galactic_reversed.transform_to(ICRS())

    map_stars_reversed, _ = get_map(NSIDE, cat_reversed.ra.value*u.deg, cat_reversed.dec.value*u.deg, 
                                        func_name='count', null_val=0)

    map_stars = get_star_map(NSIDE=NSIDE, fn_map=fn_starmap)
    map_mcs = map_stars - map_stars_reversed

    coord_lmc = SkyCoord('5h23m34.5s', '-69d45m22s', frame='icrs')
    coord_smc = SkyCoord('0h52m44.8s', '-72d49m43s', frame='icrs')

    sep_max_lmc = 9*u.deg
    sep_max_smc = 5*u.deg

    vec_lmc = hp.ang2vec(coord_lmc.ra.value, coord_lmc.dec.value, lonlat=True)
    vec_smc = hp.ang2vec(coord_smc.ra.value, coord_smc.dec.value, lonlat=True)
    # returns list of indices (not booleans)
    ipix_lmc = hp.query_disc(nside=NSIDE, vec=vec_lmc, radius=sep_max_lmc.to('radian').value)
    ipix_smc = hp.query_disc(nside=NSIDE, vec=vec_smc, radius=sep_max_smc.to('radian').value)

    i_mcs = np.full(len(map_mcs), False)
    i_mcs[ipix_lmc] = True
    i_mcs[ipix_smc] = True

    map_mcs[~i_mcs] = 0.0
    
    if fn_map is not None:
        np.save(fn_map, map_mcs)
        print(f"Saved MCs map to {fn_map}")
    return map_mcs


### Completeness / M10 selection function model map functions


def get_m10_map(NSIDE, fn_map=None):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"M10 map already exists, loading from {fn_map}")
        return np.load(fn_map)
    print(f"Generating new m10 map ({fn_map})")
    
    from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG_hpx7 
    dr3sf = DR3SelectionFunctionTCG_hpx7()
    m10_map_hpx7 = dr3sf.m10map[:,2]
    # careful with orders here! this makes it match the others when plotted w projview
    m10_map = hp.ud_grade(m10_map_hpx7, NSIDE, order_in='NESTED', order_out='RING')
    if fn_map is not None:
        np.save(fn_map, m10_map)
        print(f"Saved m10 map to {fn_map}")
    return m10_map


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



def get_dust_map(NSIDE=None, R=3.1, fn_map=None):
    if fn_map is not None and os.path.exists(fn_map):
        print(f"Dustmap already exists, loading from {fn_map}")
        return np.load(fn_map)
    assert NSIDE is not None and R is not None, f"{fn_map} doesn't exist; must pass NSIDE to generate!"
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