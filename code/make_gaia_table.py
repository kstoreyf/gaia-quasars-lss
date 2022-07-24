import numpy as np

from astropy.table import Table, join

import utils


def main():

    # save names
    fn_gaia_slim = '../data/gaia_slim.fits'
    fn_gaia_withspz = '../data/gaia_spz_kNN.fits'
    overwrite = True

    # data paths 
    fn_gaia = '../data/gaia_wise_panstarrs_tmass.fits.gz'
    fn_spz = '../data/redshifts_spz_kNN.fits'

    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    tab_gaia_slim = save_subset(tab_gaia, fn_gaia_slim, overwrite=overwrite)

    # SPZ-only table
    tab_spz = utils.load_table(fn_spz)
    save_spz_only(tab_gaia_slim, tab_spz, fn_gaia_withspz, overwrite=overwrite)


def save_subset(tab, fn_save, overwrite=False):
    utils.add_gaia_wise_colors(tab)
    utils.add_ebv(tab)
    Rv = 3.1
    Av = Rv*tab['ebv']
    tab.add_column(Av, name='A_v')
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mpro', 'w2mpro', 'allwise_oid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc']
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