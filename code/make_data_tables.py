import numpy as np

from astropy.table import Table, join

import utils


def main():
    #gaia_slim()
    #spz_only()
    sdss_slim()


def gaia_slim(overwrite=False):
    # save names
    fn_gaia_slim = '../data/gaia_slim.fits'
   
    # data paths 
    fn_gaia = '../data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mpro', 'w2mpro', 'allwise_oid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc']
    tab_gaia_slim = save_subset(tab_gaia, columns_to_keep, fn_gaia_slim, overwrite=overwrite)


def sdss_slim(overwrite=False):
    # save name
    fn_sdss_slim = '../data/sdss_slim.fits'
   
    # data paths 
    fn_sdss = '../data/SDSS_DR16Q_v4.fits'

    # Load data
    print("Loading data")
    tab_sdss = utils.load_table(fn_sdss)

    # Create and save
    columns_to_keep = ['SDSS_NAME', 'OBJID', 'RA', 'DEC', 'Z']
    save_subset(tab_sdss, columns_to_keep, fn_sdss_slim, overwrite=overwrite)


def spz_only(overwrite=False):
    # save name
    fn_gaia_withspz = '../data/gaia_spz_kNN.fits'

    # data paths
    fn_gaia_slim = '../data/gaia_slim.fits'
    fn_spz = '../data/redshifts_spz_kNN.fits'

    # load data
    tab_gaia_slim = utils.load_table(fn_gaia_slim)

    # SPZ-only table
    tab_spz = utils.load_table(fn_spz)
    save_spz_only(tab_gaia_slim, tab_spz, fn_gaia_withspz, overwrite=overwrite)


def save_subset(tab, columns_to_keep, fn_save, overwrite=False):
    gaia_wise_colors = ['g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2']
    if np.any(np.in1d(gaia_wise_colors, columns_to_keep)):
        utils.add_gaia_wise_colors(tab)
    if 'ebv' in columns_to_keep or 'A_v' in columns_to_keep:
        utils.add_ebv(tab)
    if 'A_v' in columns_to_keep:
        Rv = 3.1
        Av = Rv*tab['ebv']
        tab.add_column(Av, name='A_v')

    tab.keep_columns(columns_to_keep)
    tab.write(fn_save, overwrite=overwrite)
    print(f"Wrote table with {len(tab)} objects to {fn_save}")
    return tab


def save_spz_only(tab_main, tab_spz, fn_save, overwrite=False):
    idx_withspz = np.isfinite(tab_spz['redshift_spz'])
    tab_spz_withspz = tab_spz[idx_withspz]
    tab_joined = join(tab_main, tab_spz_withspz, keys='source_id', join_type='inner')
    tab_joined.write(fn_save, overwrite=overwrite)
    print(f"Wrote table with {len(tab_joined)} objects to {fn_save}")
    return tab_joined


if __name__=='__main__':
    main()