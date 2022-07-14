import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from dustmaps.sfd import SFDQuery


def get_fraction_recovered(Y_true, Y_hat, z_err_close):
        return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)


def add_spzs(tab_gaia):
    fn_spz = '../data/redshifts_spz_kNN.fits'
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
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs') 
    ebv = sfd(coords)
    return ebv


def get_extinction(ra, dec, R=3.1):
    ebv = get_ebv(ra, dec)
    return R*ebv



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
        #vals_by_pixel = np.add.at(map, pixel_indices, values)
        map = np.bincount(pixel_indices, weights=quantity, minlength=NPIX)
        # divide by counts to get mean
        map /= counts_by_pixel 

    # If we want to put in values other than 0 where no data,
    # need to do like this bc bincount just does 0
    if null_val != 0:
        pixel_arr = np.arange(NPIX)
        pixels_nodata = list(set(pixel_arr) - set(pixel_indices))
        map[pixels_nodata] = null_val
    
    return map, pixel_indices