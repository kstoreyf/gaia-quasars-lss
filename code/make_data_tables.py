import numpy as np

from astropy.table import Table, join
from astropy import units as u

import utils


def main():
    overwrite = False
    gaia_slim(overwrite=overwrite)

    #gaia_slim_xsdss(overwrite=overwrite)
    #gaia_clean(overwrite=overwrite)

    #G_maxs = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4]
    # G_maxs = [20.0]
    # fn_spz='../data/redshifts_spz_kNN_G20.5.fits'
    # for G_max in G_maxs:
    #     merge_gaia_spzs_and_cutGmax(fn_spz=fn_spz, G_max=G_max, overwrite=overwrite)


def gaia_slim(overwrite=False):
    # save names
    fn_gaia_slim = '../data/gaia_slim.fits'
   
    # data paths 
    fn_gaia = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)
    print(sdfkjsdf)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mpro', 'w2mpro', 'allwise_oid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc',
                        'parallax', 'parallax_error', 
                        'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']
    tab_gaia_slim = save_slim_table(tab_gaia, columns_to_keep, fn_gaia_slim, overwrite=overwrite)


def gaia_clean(overwrite=False):
    
    fn_gaia_clean = '../data/gaia_clean.fits'

    fn_gaia = '../data/gaia_slim_xsdss.fits'
    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print('N_gaia:', len(tab_gaia))

    print("Making color cuts")
    color_cuts = [[0., 1., 0.2], [1., 1., 2.9]]
    idx_clean_gaia = utils.gw1_w1w2_cuts_index(tab_gaia, color_cuts) 
    tab_gaia_clean = tab_gaia[idx_clean_gaia]
    print(len(tab_gaia_clean))

    print("N_clean:", len(tab_gaia_clean))
    tab_gaia_clean.write(fn_gaia_clean, overwrite=overwrite)


def sdss_slim(overwrite=False):
    # save name
    fn_sdss_slim = '../data/sdss_slim.fits'
   
    # data paths 
    fn_sdss = '../data/SDSS_DR16Q_v4.fits'

    # Load data
    print("Loading data")
    tab_sdss = utils.load_table(fn_sdss)

    # Create and save
    columns_to_keep = ['SDSS_NAME', 'OBJID', 'THING_ID', 'RA', 'DEC', 'Z']
    save_slim_table(tab_sdss, columns_to_keep, fn_sdss_slim, overwrite=overwrite)


def gaia_slim_xsdss(overwrite=False):

    fn_gaia_slim_xsdss = '../data/gaia_slim_xsdss.fits'
    
    # Load data
    print("Loading gaia data")
    fn_gaia_slim = '../data/gaia_slim.fits'
    tab_gaia = utils.load_table(fn_gaia_slim)
    print('N_gaia:', len(tab_gaia))

    print("Load in SDSS data")
    fn_sdss_slim = '../data/sdss_slim.fits'
    tab_sdss = utils.load_table(fn_sdss_slim)

    # Clean out super low redshift SDSS objects
    # TODO: double check if should be doing this
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'Z'
    idx_zgood = utils.redshift_cut_index(tab_sdss, z_min, redshift_key)
    tab_sdss = tab_sdss[idx_zgood]
    print(f"Number of SDSS QSOs: {len(tab_sdss)}")

    # Cross-match
    print("Performing cross-match")
    separation = 1*u.arcsec
    index_list_gaiaINsdss, index_list_sdssINgaia = utils.cross_match(
                                           tab_gaia['ra'], tab_gaia['dec'],
                                           tab_sdss['RA']*u.degree, tab_sdss['DEC']*u.degree,
                                           separation=separation)

    print("N_gaiaINsdss:", len(index_list_gaiaINsdss))
    print("N_sdssINgaia:", len(index_list_sdssINgaia))
                               
    # check this line!
    sdss_column_names = ['SDSS_NAME', 'OBJID', 'THING_ID', 'RA', 'DEC', 'Z']
    for sdss_column_name in sdss_column_names:
        column_name = 'sdss_'+sdss_column_name
        column = np.full(len(tab_gaia), np.nan)
        # check this line!
        column[index_list_gaiaINsdss] = tab_sdss[index_list_sdssINgaia]['Z']
        tab_gaia[column_name] = column 

    print("done, saving!")
    tab_gaia.write(fn_gaia_slim_xsdss, overwrite=overwrite)


def merge_gaia_spzs_and_cutGmax(fn_spz='../data/redshifts_spz_kNN_G20.5.fits',
                                G_max=20.5, overwrite=False):

    # save name
    fn_gaia_withspz = f'../data/gaia_spz_G{G_max}.fits'

    # data paths
    fn_gaia = '../data/gaia_clean.fits'
    assert G_max <= 20.5, "SPZs only go to 20.5!"

    # load data, cut to G_max
    tab_gaia = utils.load_table(fn_gaia)
    tab_gaia = tab_gaia[tab_gaia['phot_g_mean_mag'] < G_max]

    # SPZ-only table
    tab_spz = utils.load_table(fn_spz)
    join_and_save(tab_gaia, tab_spz, fn_gaia_withspz, overwrite=overwrite)


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


def join_and_save(tab1, tab2, fn_save, join_key='source_id', overwrite=False):
    tab_joined = join(tab1, tab2, keys=join_key, join_type='inner')
    tab_joined.write(fn_save, overwrite=overwrite)
    print(f"Wrote table with {len(tab_joined)} objects to {fn_save}")
    return tab_joined


if __name__=='__main__':
    main()
