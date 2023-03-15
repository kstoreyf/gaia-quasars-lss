import numpy as np

from astropy.table import Table, join
from astropy import units as u

import utils


def main():
    overwrite = True
    #gaia_slim(overwrite=overwrite)
    #sdss_slim(overwrite=overwrite)
    #quasars_sdss_xgaia_good(overwrite=overwrite)
    #galaxies_sdss_xgaia_good(overwrite=overwrite)
    #stars_sdss_xgaia_good(overwrite=overwrite)
    remove_duplicate_sources(overwrite=overwrite)

    #gaia_unwise_slim(overwrite=overwrite)
    #gaia_catwise_slim(overwrite=overwrite)

    #gaia_slim_xsdss(overwrite=overwrite)
    #gaia_clean(overwrite=overwrite)

    #G_maxs = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4]
    # G_maxs = [20.0, 20.4]
    # fn_spz='../data/redshift_estimates/redshifts_spz_kNN_G20.5_regression.fits'
    # for G_max in G_maxs:
    #     merge_gaia_spzs_and_cutGmax(fn_spz=fn_spz, G_max=G_max, overwrite=overwrite)

    # save as csv
    # fn_gaia_slim = '../data/gaia_slim.fits'
    # tab_gaia = utils.load_table(fn_gaia_slim)
    # fn_csv = '../data/gaia_candidates.csv'
    # save_as_csv(tab_gaia, ['source_id', 'ra', 'dec'], fn_csv, overwrite=overwrite)


def gaia_slim(overwrite=False):
    # save names
    fn_gaia_slim = '../data/gaia_slim.fits'
   
    # data paths 
    fn_gaia = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)

    rng = np.random.default_rng(seed=42)
    tab_gaia['rand_ints'] = rng.choice(range(len(tab_gaia)), size=len(tab_gaia), replace=False)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mpro', 'w2mpro', 'allwise_oid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc',
                        'parallax', 'parallax_error', 
                        'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'rand_ints']
    tab_gaia_slim = save_slim_table(tab_gaia, columns_to_keep, fn_gaia_slim, overwrite=overwrite)


def gaia_unwise_slim(overwrite=False):
    # save names
    fn_gaia_unwise_slim = '../data/gaia_unwise_slim.fits'
   
    # data paths 
    fn_gaia = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)

    fn_xwise = '../data/gaia_candidates_unwise_nn.csv'
    tab_xwise = Table.read(fn_xwise, format='csv')
    tab_xwise.keep_columns(['t1_source_id', 'mag_w1_vg', 'mag_w2_vg', 'unwise_objid'])
    print(tab_xwise.columns)
    tab_gaia_xwise = join(tab_gaia, tab_xwise, keys_left='source_id', keys_right='t1_source_id',
                          join_type='left')
    print(tab_gaia_xwise.columns)

    rng = np.random.default_rng(seed=42)
    tab_gaia_xwise['rand_ints'] = rng.choice(range(len(tab_gaia_xwise)), size=len(tab_gaia_xwise), replace=False)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'mag_w1_vg', 'mag_w2_vg', 'unwise_objid',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc',
                        'parallax', 'parallax_error', 
                        'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'rand_ints']
    tab_gaia_xwise_slim = save_slim_table(tab_gaia_xwise, columns_to_keep, fn_gaia_unwise_slim, overwrite=overwrite)


def gaia_catwise_slim(overwrite=False):
    # save names
    fn_gaia_unwise_slim = '../data/gaia_catwise_slim.fits'
   
    # data paths 
    fn_gaia = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)

    fn_xwise = '../data/gaia_candidates_catwise_nn.csv'
    tab_xwise = Table.read(fn_xwise, format='csv')
    print(tab_xwise.columns)
    tab_xwise.keep_columns(['t1_source_id', 'source_id', 'w1mag', 'w2mag'])
    tab_xwise.rename_column('source_id', 'catwise_source_id')
    tab_gaia_xwise = join(tab_gaia, tab_xwise, keys_left='source_id', keys_right='t1_source_id',
                          join_type='left')
    print(tab_gaia_xwise.columns)

    rng = np.random.default_rng(seed=42)
    tab_gaia_xwise['rand_ints'] = rng.choice(range(len(tab_gaia_xwise)), size=len(tab_gaia_xwise), replace=False)

    # Create and save
    columns_to_keep = ['source_id', 'ra', 'dec', 'l', 'b', 'redshift_qsoc', 'ebv', 'A_v',
                        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                        'w1mag', 'w2mag', 'catwise_source_id',
                        'redshift_qsoc_lower','redshift_qsoc_upper','zscore_qsoc','flags_qsoc',
                        'parallax', 'parallax_error', 
                        'pm', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'rand_ints']
    tab_gaia_xwise_slim = save_slim_table(tab_gaia_xwise, columns_to_keep, fn_gaia_unwise_slim, 
                                          w1_name='w1mag', w2_name='w2mag', overwrite=overwrite)




def gaia_clean(overwrite=False):
    
    fn_gaia_clean = '../data/gaia_clean.fits'

    fn_gaia = '../data/gaia_slim_xsdss.fits'
    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print('N_gaia:', len(tab_gaia))

    print("Cutting sources without all necessary data (colors and QSOC redshifts)")
    # TODO is there a reason i shouldnt be cutting on QSOC redshift here? or having g,bp,rp,w1,w2?
    col_names_necessary = ['redshift_qsoc', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'w1mpro', 'w2mpro']
    cols_necessary = []
    for col_name in col_names_necessary:
        cols_necessary.append(tab_gaia[col_name])
    cols_necessary = np.array(cols_necessary).T
    i_has_necessary = np.all(np.isfinite(cols_necessary), axis=1)
    print(f"Has necessary color data: {np.sum(i_has_necessary)} ({np.sum(i_has_necessary)/len(i_has_necessary):.3f})")
    tab_gaia_nec = tab_gaia[i_has_necessary]

    print("Making color cuts")
    color_cuts = [[0., 1., 0.2], [1., 1., 2.9]]
    idx_clean_gaia = utils.gw1_w1w2_cuts_index(tab_gaia_nec, color_cuts) 
    tab_gaia_clean = tab_gaia_nec[idx_clean_gaia]
    print(len(tab_gaia_clean))

    rng = np.random.default_rng(seed=42)
    tab_gaia_clean['rand_ints_clean'] = rng.choice(range(len(tab_gaia_clean)), size=len(tab_gaia_clean), replace=False)

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
    columns_to_keep = ['SDSS_NAME', 'OBJID', 'THING_ID', 'RA', 'DEC', 'Z', 'ZWARNING']
    save_slim_table(tab_sdss, columns_to_keep, fn_sdss_slim, overwrite=overwrite)


def gaia_slim_xsdss(fn_gaia_slim, fn_gaia_slim_xsdss, overwrite=False):

    #fn_gaia_slim_xsdss='../data/gaia_slim_xsdss.fits'
    
    # Load data
    print("Loading gaia data")
    #fn_gaia_slim = '../data/gaia_slim.fits'
    tab_gaia = utils.load_table(fn_gaia_slim)
    print('N_gaia:', len(tab_gaia))

    print("Load in SDSS data")
    fn_sdss_slim = '../data/sdss_slim.fits'
    tab_sdss = utils.load_table(fn_sdss_slim)
    print(f"Number of SDSS QSOs: {len(tab_sdss)}")


    # Clean out super low redshift SDSS objects, and ones with bad redshifts
    # TODO: double check if should be doing this
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'Z'
    idx_zgood = utils.redshift_cut_index(tab_sdss, z_min, redshift_key)
    tab_sdss = tab_sdss[idx_zgood]

    # https://www.sdss4.org/dr16/algorithms/bitmasks/#ZWARNING
    idx_zwarn0 = tab_sdss['ZWARNING']==0
    tab_sdss = tab_sdss[idx_zwarn0]
    print(f"Number of SDSS QSOs with good redshfits: {len(tab_sdss)}")


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


def quasars_sdss_xgaia_good(overwrite=False):

    #fn_sdss_xgaia_good = '../data/sdss_xgaia_wise_good.fits.gz'
    fn_sdss_xgaia_good = '../data/quasars_sdss_xgaia_xunwise_good.fits'

    print("Load in SDSS xgaia data")
    fn_sdss_xgaia = '../data/quasars_sdss_xgaia_xunwise.csv'
    tab_sdss_xgaia = utils.load_table(fn_sdss_xgaia, format='csv')
    print(f"Number of SDSS xGaia QSOs: {len(tab_sdss_xgaia)}")
    tab_sdss_xgaia.rename_column('ra', 'ra_unwise')
    tab_sdss_xgaia.rename_column('dec', 'dec_unwise')
    for column_name in list(tab_sdss_xgaia.columns):
        new_name = column_name
        if column_name.startswith('t1'):
            new_name = column_name.split('t1_')[-1]
            if new_name=='z':
                new_name = 'z_sdss'
        tab_sdss_xgaia.rename_column(column_name, new_name)

    # Keep sources with phot_bp_n_obs and phot_rp_n_obs >= 5
    i_good_nobs = (tab_sdss_xgaia['phot_bp_n_obs'] >= 5) & (tab_sdss_xgaia['phot_rp_n_obs'] >= 5)
    print(f"Removing {np.sum(~i_good_nobs)} sources with <5 bp or rp n_obs")
    tab_sdss_xgaia = tab_sdss_xgaia[i_good_nobs]

    # Clean out super low redshift SDSS objects, and ones with bad redshifts
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'z_sdss'
    idx_zgood = utils.redshift_cut_index(tab_sdss_xgaia, z_min, redshift_key)
    print(f"Removing {np.sum(~idx_zgood)} sources with z<{z_min}")
    tab_sdss_xgaia = tab_sdss_xgaia[idx_zgood]
    print(f"Number of SDSS QSOs with good redshfits: {len(tab_sdss_xgaia)}")
    # Note that we already did zwarning cut in Gaia cross-match, so don't need to here (didn't save zwarning)

    tab_sdss_xgaia.write(fn_sdss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_sdss_xgaia)} objects to {fn_sdss_xgaia_good}")


# galaxies via sdss skyserver CAS, 
# https://skyserver.sdss.org/CasJobs/jobdetails.aspx?id=59558048&message=Details%20of%2059558048
# SELECT
# specObjID, ra, dec into mydb.MyTable from SpecObj
# WHERE class='GALAXY' AND subClass!='AGN' AND subClass!='AGN BROADLINE' AND zWarning=0
def galaxies_sdss_xgaia_good(overwrite=False):

    fn_gals_sdss_xgaia_good = '../data/galaxies_sdss_xgaia_xunwise_good.fits'

    print("Load in SDSS xgaia data")
    fn_gals_sdss_xgaia = '../data/galaxies_sdss_xgaia_xunwise.csv'
    tab_gals_sdss_xgaia = utils.load_table(fn_gals_sdss_xgaia, format='csv')
    print(f"Number of SDSS xGaia Galaxies: {len(tab_gals_sdss_xgaia)}")
    print(tab_gals_sdss_xgaia.columns)
    tab_gals_sdss_xgaia.rename_column('ra', 'ra_unwise')
    tab_gals_sdss_xgaia.rename_column('dec', 'dec_unwise')
    for column_name in list(tab_gals_sdss_xgaia.columns):
        new_name = column_name
        if column_name.startswith('t1'):
            new_name = column_name.split('t1_')[-1]
        tab_gals_sdss_xgaia.rename_column(column_name, new_name)
    print(tab_gals_sdss_xgaia.columns)
       
    # We already removed ZWARNING gals in SQL query on SDSS archive!

    # Keep sources with phot_bp_n_obs and phot_rp_n_obs >= 5
    i_good_nobs = (tab_gals_sdss_xgaia['phot_bp_n_obs'] >= 5) & (tab_gals_sdss_xgaia['phot_rp_n_obs'] >= 5)
    print(f"Removing {np.sum(~i_good_nobs)} sources with <5 bp or rp n_obs")
    tab_gals_sdss_xgaia = tab_gals_sdss_xgaia[i_good_nobs]

    # Remove apparent stellar contaminants (Bailer-Jones 2019)
    G = tab_gals_sdss_xgaia['phot_g_mean_mag']
    BP = tab_gals_sdss_xgaia['phot_bp_mean_mag']
    RP = tab_gals_sdss_xgaia['phot_rp_mean_mag']
    i_badgal = (G - RP) < (0.3 + 1.1*(BP-G) - 0.29*(BP-G)**2)
    print(f"Removing {np.sum(i_badgal)} suspected stellar contaminants")
    tab_gals_sdss_xgaia = tab_gals_sdss_xgaia[~i_badgal]

    tab_gals_sdss_xgaia.write(fn_gals_sdss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gals_sdss_xgaia)} objects to {fn_gals_sdss_xgaia_good}")


def stars_sdss_xgaia_good(overwrite=False):

    fn_stars_sdss_xgaia_good = '../data/stars_sdss_xgaia_xunwise_good.fits'

    print("Load in SDSS xgaia data")
    fn_stars_sdss_xgaia = '../data/stars_sdss_xgaia_xunwise.csv'
    tab_stars_sdss_xgaia = utils.load_table(fn_stars_sdss_xgaia, format='csv')
    print(f"Number of SDSS xGaia Stars: {len(tab_stars_sdss_xgaia)}")
    print(tab_stars_sdss_xgaia.columns)
    tab_stars_sdss_xgaia.rename_column('ra', 'ra_unwise')
    tab_stars_sdss_xgaia.rename_column('dec', 'dec_unwise')
    for column_name in list(tab_stars_sdss_xgaia.columns):
        new_name = column_name
        if column_name.startswith('t1'):
            new_name = column_name.split('t1_')[-1]
        tab_stars_sdss_xgaia.rename_column(column_name, new_name)
    print(tab_stars_sdss_xgaia.columns)
       
    # We already removed ZWARNING gals in SQL query on SDSS archive!

    # Keep sources with phot_bp_n_obs and phot_rp_n_obs >= 5
    i_good_nobs = (tab_stars_sdss_xgaia['phot_bp_n_obs'] >= 5) & (tab_stars_sdss_xgaia['phot_rp_n_obs'] >= 5)
    print(f"Removing {np.sum(~i_good_nobs)} sources with <5 bp or rp n_obs")
    tab_stars_sdss_xgaia = tab_stars_sdss_xgaia[i_good_nobs]

    tab_stars_sdss_xgaia.write(fn_stars_sdss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_stars_sdss_xgaia)} objects to {fn_stars_sdss_xgaia_good}")


def remove_duplicate_sources(overwrite=False):

    fn_quasars = '../data/quasars_sdss_xgaia_xunwise_good.fits'
    fn_galaxies = '../data/galaxies_sdss_xgaia_xunwise_good.fits'
    fn_stars = '../data/stars_sdss_xgaia_xunwise_good.fits'
    fns = [fn_quasars, fn_galaxies, fn_stars]
    source_ids = []
    tabs = []
    for fn in fns:
        tab = utils.load_table(fn)
        source_ids.extend(list(tab['source_id']))
        tabs.append(tab)
        # u, c = np.unique(list(tab['source_id']), return_counts=True)
        # dup = u[c > 1]
        # print(len(dup))
        # print(dup)

    # This finds duplicates both within tables and across tables, as we want
    u, c = np.unique(source_ids, return_counts=True)
    dup = u[c > 1]
    print(len(dup))

    for tab in tabs:
        i_nondup = 
    #for tab in tabs:
        
    


def save_slim_table(tab, columns_to_keep, fn_save, overwrite=False, 
                    w1_name='mag_w1_vg', w2_name='mag_w2_vg'):
    gaia_wise_colors = ['g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2']
    if np.any(np.in1d(gaia_wise_colors, columns_to_keep)):
        utils.add_gaia_wise_colors(tab, w1_name=w1_name, w2_name=w2_name)
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


def save_as_csv(tab, column_names, fn_csv, overwrite=False):
    tab = tab.copy()
    tab.keep_columns(column_names)
    tab.write(fn_csv, format='csv', overwrite=overwrite)  
    print(f"Saved table as CSV to {fn_csv}!")



if __name__=='__main__':
    main()
