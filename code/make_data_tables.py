import numpy as np

from astropy.table import Table, join

import utils


def main():
    overwrite = True
    #gaia_slim(overwrite=overwrite)
    #G_maxs = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4]
    G_maxs = [20.0]
    for G_max in G_maxs:
        gaia_subset(G_max=G_max, overwrite=overwrite)


def gaia_slim(overwrite=False):
    # save names
    fn_gaia_slim = '../data/gaia_slim.fits'
   
    # data paths 
    fn_gaia = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mpro', 'w2mpro', 'allwise_oid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc',
                        'parallax', 'parallax_error', 
                        'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']
    tab_gaia_slim = save_slim_table(tab_gaia, columns_to_keep, fn_gaia_slim, overwrite=overwrite)


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
    save_slim_table(tab_sdss, columns_to_keep, fn_sdss_slim, overwrite=overwrite)


def gaia_subset(G_max, overwrite=False):
    # save name
    fn_gaia_withspz = f'../data/gaia_G{G_max}.fits'

    # data paths
    fn_gaia_slim = '../data/gaia_slim.fits'
    assert G_max < 20.5, "SPZs only go to 20.5!"
    fn_spz = '../data/redshifts_spz_kNN_G20.5.fits'

    # load data, cut to G_max
    tab_gaia_slim = utils.load_table(fn_gaia_slim)
    tab_gaia_slim = tab_gaia_slim[tab_gaia_slim['phot_g_mean_mag'] < G_max]

    # SPZ-only table
    # The ones that have SPZs have already gone through the color cut, so this effectively
    # cuts to color, as well as saves SPZs. 
    # But TODO: probs should do all this in a way that makes that clearer / decouples those tables
    tab_spz = utils.load_table(fn_spz)
    save_subset_with_spzs(tab_gaia_slim, tab_spz, fn_gaia_withspz, overwrite=overwrite)


def save_slim_table(tab, columns_to_keep, fn_save, overwrite=False):
    gaia_wise_colors = ['g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2']
    if np.any(np.in1d(gaia_wise_colors, columns_to_keep)):
        utils.add_gaia_wise_colors(tab)
    if 'ebv' in columns_to_keep or 'A_v' in columns_to_keep:
        utils.add_ebv(tab)
    if 'A_v' in columns_to_keep:
        Rv = 3.1
        Av = Rv*tab['ebv']
        tab.add_column(Av, name='A_v')
    if 'pm' in columns_to_keep:
        pm = np.sqrt(tab['pmra']**2 + tab['pmdec']**2)
        tab.add_column(pm, name='pm')
    tab.keep_columns(columns_to_keep)
    tab.write(fn_save, overwrite=overwrite)
    print(f"Wrote table with {len(tab)} objects to {fn_save}")
    return tab


def save_subset_with_spzs(tab_main, tab_spz, fn_save, overwrite=False):
    idx_withspz = np.isfinite(tab_spz['redshift_spz'])
    tab_spz_withspz = tab_spz[idx_withspz]
    tab_joined = join(tab_main, tab_spz_withspz, keys='source_id', join_type='inner')
    tab_joined.write(fn_save, overwrite=overwrite)
    print(f"Wrote table with {len(tab_joined)} objects to {fn_save}")
    return tab_joined


if __name__=='__main__':
    main()
