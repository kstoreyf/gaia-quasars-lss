# **************************************************
# * File Name : write_simple_selfunc.py
# * Creation Date : 2025-07-30
# * Created By : kstoreyf
# * Description :
# **************************************************
import healpy as hp
import numpy as np

# Set NSIDE
nside = 64

# Create map of all ones
npix = hp.nside2npix(nside)
ones_map = np.ones(npix)

# Write to FITS file
hp.write_map("selection_function_NSIDE64_ones.fits", ones_map, overwrite=True)

print(f"Created selection_function_NSIDE64_ones.fits with NSIDE={nside} ({npix} pixels)")
