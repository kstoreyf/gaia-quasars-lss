import numpy as np

import healpy as hp
#from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, SkyCoord
from astropy import units as u

import utils


def main():
    pass


def galactic_plane_mask(NSIDE, b_max, fn_mask=None):
    NPIX = hp.nside2npix(NSIDE)
    mask = np.ones(NPIX, dtype=bool)
    #hpa = HEALPix(nside=NSIDE, frame=Galactic())
    #coords = hpa.healpix_to_skycoord(np.arange(NPIX))  
    #bs = coords['b']
    ra, dec = hp.pix2ang(NSIDE, np.arange(NPIX), lonlat=True)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    bs = coords.galactic.b
    idx_keep = np.abs(bs.value) > b_max
    mask[idx_keep] = 0
    if fn_mask is not None:
        hp.write_map(mask, fn_mask)
    return mask



def magellanic_clouds_mask(NSIDE, fn_mask=None):
    NPIX = hp.nside2npix(NSIDE)
    # start mask with 0s, meaning kee[]
    mask = np.zeros(NPIX, dtype=bool)

    coord_lmc = SkyCoord('5h23m34.5s', '-69d45m22s', frame='icrs')
    coord_smc = SkyCoord('0h52m44.8s', '-72d49m43s', frame='icrs')

    sep_max_lmc = 6*u.deg
    sep_max_smc = 3*u.deg

    vec_lmc = hp.ang2vec(coord_lmc.ra.value, coord_lmc.dec.value, lonlat=True)
    vec_smc = hp.ang2vec(coord_smc.ra.value, coord_smc.dec.value, lonlat=True)
    # returns list of indices (not booleans)
    ipix_nolmc = hp.query_disc(nside=NSIDE, vec=vec_lmc, radius=sep_max_lmc.to('radian').value)
    ipix_nosmc = hp.query_disc(nside=NSIDE, vec=vec_smc, radius=sep_max_smc.to('radian').value)
    # 1s mean masked (excluded) vals
    mask[ipix_nolmc] = 1 
    mask[ipix_nosmc] = 1
    if fn_mask is not None:
        hp.write_map(mask, fn_mask)
    return mask


def galactic_dust_mask(NSIDE, Av_max, R, rng, fn_mask=None):
    map_Av = utils.get_dust_map(NSIDE, rng, R=R)
    mask = map_Av > Av_max 
    if fn_mask is not None:
        hp.write_map(mask, fn_mask)
    return mask


def subsample_by_mask(NSIDE, ra_rand, dec_rand, mask_func, mask_func_args):
    mask = mask_func(NSIDE, *mask_func_args)
    _, pixel_indices = utils.get_map(NSIDE, ra_rand, dec_rand)
        
    # TODO: better way to do this??
    pixel_arr = np.arange(len(mask))
    pixel_indices_keep = pixel_arr[mask]
    idx_keep = np.in1d(pixel_indices, pixel_indices_keep, invert=True)
    print(f"Applied mask, kept {np.sum(idx_keep)/len(idx_keep):.3f} of sources")
    return ra_rand[idx_keep], dec_rand[idx_keep]


if __name__=='__main__':
    main()