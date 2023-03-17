import numpy as np

from astropy.table import Table, join, vstack
from astropy import units as u

import utils


def main():
    overwrite = True
    #gaia_candidates_plus_info(overwrite=overwrite)
    #gaia_candidates_xunwise_good(overwrite=overwrite)
    #make_labeled_table(overwrite=overwrite)

    #gaia_slim(overwrite=overwrite)
    #sdss_slim(overwrite=overwrite)
    #gaia_purer_sourceids(overwrite=overwrite)

    # quasars_sdss_xgaia_good(overwrite=overwrite)
    # galaxies_sdss_xgaia_good(overwrite=overwrite)
    # stars_sdss_xgaia_good(overwrite=overwrite)
    # remove_duplicate_sources(overwrite=overwrite)
    get_gaia_xsdssfootprint(overwrite=overwrite)

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



def gaia_purer_sourceids(overwrite=False):
    fn_gaia_purer = '/scratch/ksf293/gaia-quasars-lss/data/gaia_purer.fits'
    fn_gaia_purer_sourceids = '../data/gaia_purer_sourceids.fits'
    tab_gaia_purer = utils.load_table(fn_gaia_purer)
    tab_gaia_purer.keep_columns(['source_id'])
    tab_gaia_purer.write(fn_gaia_purer_sourceids, overwrite=overwrite)


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
    tab_gaia_nec = get_table_with_necessary(tab_gaia)

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


def add_randints_column(tab):
    rng = np.random.default_rng(seed=42)
    tab['rand_ints'] = rng.choice(range(len(tab)), size=len(tab), replace=False)


def gaia_candidates_plus_info(overwrite=False):

    # save to:
    fn_gaia_plus = '../data/gaia_candidates_plus.fits.gz'

    fn_gaia = '../data/gaia_candidates.fits.gz'
    tab_gaia = utils.load_table(fn_gaia)

    utils.add_ebv(tab_gaia)
    Rv = 3.1
    Av = Rv*tab_gaia['ebv']
    tab_gaia.add_column(Av, name='A_v')

    pm = np.sqrt(tab_gaia['pmra']**2 + tab_gaia['pmdec']**2)
    tab_gaia.add_column(pm, name='pm')

    add_randints_column(tab_gaia)

    tab_gaia.write(fn_gaia_plus, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gaia)} objects to {fn_gaia_plus}")


def gaia_candidates_xunwise_good(overwrite=False):

    # good will have wise in it but contain all rows.
    fn_gaia_xwise = '../data/gaia_candidates_wnec.fits'

    # data paths 
    fn_gaia = '../data/gaia_candidates_plus.fits.gz'
    fn_xwise = '../data/gaia_candidates_xunwise_all.csv'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(tab_gaia.columns)
    tab_xwise = Table.read(fn_xwise, format='csv')
    print(tab_xwise.columns)

    tab_xwise.keep_columns(['t1_source_id', 'mag_w1_vg', 'mag_w2_vg', 'unwise_objid'])
    tab_gaia_xwise = join(tab_gaia, tab_xwise, keys_left='source_id', keys_right='t1_source_id',
                          join_type='left')
    tab_gaia_xwise.remove_column('t1_source_id')                  
    print(tab_gaia_xwise.columns)

    # Require finite photometry and redshift_qsoc
    col_names_necessary = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                           'mag_w1_vg', 'mag_w2_vg', 'redshift_qsoc']
    tab_gaia_xwise = utils.get_table_with_necessary(tab_gaia_xwise, col_names_necessary=col_names_necessary)

    add_randints_column(tab_gaia_xwise)

    tab_gaia_xwise.write(fn_gaia_xwise, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gaia_xwise)} objects to {fn_gaia_xwise}")



def quasars_sdss_xgaia_good(overwrite=False):

    fn_sdss_xgaia_good = '../data/quasars_sdss_xgaia_xunwise_good.fits'

    print("Load in SDSS xgaia data")
    fn_sdss_xgaia = '../data/quasars_sdss_xgaia_xunwise_all.csv'
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
    fn_gals_sdss_xgaia = '../data/galaxies_sdss_xgaia_xunwise_all.csv'
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
    fn_stars_sdss_xgaia = '../data/stars_sdss_xgaia_xunwise_all.csv'
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

    print("Loading tables")
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

    print("Finding duplicates")
    # This finds duplicates both within tables and across tables, as we want
    # https://stackoverflow.com/a/51297779
    u, c = np.unique(source_ids, return_counts=True)
    source_ids_duplicate = u[c > 1]
    print(f"Found {len(source_ids_duplicate)} duplicated source_ids")

    for i, tab in enumerate(tabs):
        i_dup = np.isin(tab['source_id'], source_ids_duplicate)
        print(f"Removing {np.sum(i_dup)} from {fns[i]}")
        print(f"Old table size: {len(tab)}")
        tab = tab[~i_dup]
        print(f"New table size: {len(tab)}")
        fn_save = fns[i].split('.fits')[0] + '_nodup.fits'
        print(f"Saving to {fn_save}")
        tab.write(fn_save, overwrite=overwrite)        



def make_labeled_table(overwrite=False):

    # save to:
    fn_labeled = '../data/labeled_wnec.fits'

    # Requiring labeled data to be in set we will apply to, this wnec one:
    fn_gwnec = '../data/gaia_candidates_wnec.fits'
    tab_gwnec = utils.load_table(fn_gwnec)

    # Our labels come from SDSS
    tab_squasars = utils.load_table(f'../data/quasars_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS quasars: {len(tab_squasars)}")

    tab_sstars = utils.load_table(f'../data/stars_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS stars: {len(tab_sstars)}")

    tab_sgals = utils.load_table(f'../data/galaxies_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS galaxies: {len(tab_sgals)}")

    ## Stack into single table & arrays
    class_labels = ['q', 's', 'g']
    tab_squasars['class'] = 'q'
    tab_sstars['class'] = 's'
    tab_sgals['class'] = 'g'

    cols_tokeep = ['source_id', 'class', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                'mag_w1_vg', 'mag_w2_vg']

    tab_squasars.keep_columns(cols_tokeep)
    tab_sstars.keep_columns(cols_tokeep)
    tab_sgals.keep_columns(cols_tokeep)

    tab_labeled = vstack([tab_squasars, tab_sstars, tab_sgals], metadata_conflicts='silent')

    # Now that I'm only using the labeled data in wnec, i didn't need to do separate xgaia and xwise 
    # cross-matches :/ could have just crossmatched SDSS data to QSO sample. 
    # We matched the SDSS samples on their gaia match's RA and dec, so the wise properties 
    # are guaranteed to be the same as the gaia candidates 
    i_inwnec = np.isin(tab_labeled['source_id'], tab_gwnec['source_id'])
    print(f"{np.sum(i_inwnec)} of labeled data in wnec sample (out of {len(tab_labeled)}); keeping those only")
    tab_labeled = tab_labeled[i_inwnec]

    # The ones that make the i_inwnec cut will necessarily already have all the necessary data 
    # so don't need to do this
    # Require finite values of colors, and have     
    # col_names_necessary = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
    #                        'mag_w1_vg', 'mag_w2_vg']
    # tab_labeled = utils.get_table_with_necessary(tab_stars_sdss_xgaia, col_names_necessary=col_names_necessary)

    add_randints_column(tab_labeled)

    #i_train, i_val, i_test = split_train_val_test(random_ints, frac_train=0.5, frac_test=0.25, frac_val=0.25)
    #tab_labeled['']

    tab_labeled.write(fn_labeled, overwrite=overwrite)



def get_gaia_xsdssfootprint(overwrite=False):

    # save to:
    fn_xsdssfootprint = '../data/gaia_candidates_xsdssfootprint.fits'

    print("Loading tables")
    # sdss quasars (should this be only ones in wnec or something??)
    fn_squasars = '../data/sdss_slim.fits'
    tab_squasars = utils.load_table(fn_squasars)

    # full gaia candidates:
    fn_gcand = '../data/gaia_candidates_plus.fits.gz'
    tab_gcand = utils.load_table(fn_gcand)

    separation = 2*u.arcmin

    # idx_small = np.arange(75000)
    # ra_squasars = tab_squasars['RA'][idx_small]*u.deg
    # dec_squasars = tab_squasars['DEC'][idx_small]*u.deg

    ra_squasars = tab_squasars['RA']*u.deg
    dec_squasars = tab_squasars['DEC']*u.deg
    ra_gcand = tab_gcand['ra']
    dec_gcand = tab_gcand['dec']

    print("Performing cross-match")
    # for decontamination, don't care about star vs gal, label all as other 'o'
    index_list_1in2, index_list_2in1 = utils.cross_match(ra_squasars, dec_squasars,
                                                         ra_gcand, dec_gcand,
                                                         separation=separation)

    index_list_2in1_unique = np.unique(index_list_2in1)
    print(f'Found {len(index_list_2in1)} Gaia quasar candidates in range of SDSS quasars')
    print(f'{len(index_list_2in1_unique)} of these are unique')

    tab_gcand_xsdssfootprint = tab_gcand[index_list_2in1_unique]

    tab_gcand_xsdssfootprint.write(fn_xsdssfootprint, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gcand_xsdssfootprint)} objects to {fn_xsdssfootprint}")



def make_labeled_spz_set():
    pass



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
