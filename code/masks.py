import numpy as np

import healpy as hp
#from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, SkyCoord
from astropy import units as u


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


def magellanic_clouds_mask(NSIDE, fn_mask):
    pass



if __name__=='__main__':
    main()