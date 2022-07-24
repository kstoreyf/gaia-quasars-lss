import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


# Cross match function using astropy
def cross_match(ra1, dec1, ra2, dec2, separation):
    coords1 = SkyCoord(ra=ra1, dec=dec1, frame='icrs')    
    coords2 = SkyCoord(ra=ra2, dec=dec2, frame='icrs') 
    cross = astropy.coordinates.search_around_sky(coords1, coords2, separation) 
    index_list_1in2, index_list_2in1 = cross[0], cross[1] 
    return index_list_1in2, index_list_2in1

# Load in data
fn_gaia = '../data/gaia_slim.fits'
tab_gaia = Table.read(fn_gaia, format='fits')

fn_sdss = '../data/SDSS_DR16Q_v4.fits'
tab_sdss = Table.read(fn_sdss, format='fits')

# Perform cross-match; 1 arcsec is a reasonable separation
separation = 1*u.arcsec
index_list_gaiaINsdss, index_list_sdssINgaia = cross_match(tab_gaia['ra'], tab_gaia['dec'],
                                                           tab_sdss['RA']*u.degree, tab_sdss['DEC']*u.degree,
                                                           separation=separation)
tab_gaia_with_sdss_match = tab_gaia[index_list_gaiaINsdss]
tab_sdss_with_gaia_match = tab_sdss[index_list_sdssINgaia]

# Can add info from SDSS table to Gaia table, as these should have the same length now, e.g.:
tab_gaia_with_sdss_match.add_column(tab_sdss_with_gaia_match['Z'], name='redshift_sdss')

# Print number in each table; last two lines should be identical
print(f'Number of Gaia quasars: {len(tab_gaia)}')
print(f'Number of SDSS quasars: {len(tab_sdss)}')
print(f'Number of Gaia quasars with SDSS match: {len(tab_gaia_with_sdss_match)}')
print(f'Number of SDSS quasars with Gaia match: {len(tab_sdss_with_gaia_match)}')