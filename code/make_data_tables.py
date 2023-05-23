import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack

import utils


def main():
    overwrite = True
    #gaia_candidates_plus_info(overwrite=overwrite)
    #gaia_candidates_superset(overwrite=overwrite)
    ###gaia_candidates_clean(overwrite=overwrite)

    #gaia_slim(overwrite=overwrite)
    #sdss_slim(overwrite=overwrite)
    #gaia_purer_sourceids(overwrite=overwrite)

    #quasars_sdss_xgaia_good(overwrite=overwrite)
    #quasars_sdss_xgaiaall_good(overwrite=overwrite)
    # galaxies_sdss_xgaia_good(overwrite=overwrite)
    # stars_sdss_xgaia_good(overwrite=overwrite)
    #mcs_xgaia(overwrite=overwrite)
    #remove_duplicate_sources(overwrite=overwrite)

    #make_labeled_table(overwrite=overwrite)
    
    #get_gaia_xsdssfootprint(overwrite=overwrite)

    #gaia_unwise_slim(overwrite=overwrite)
    #gaia_catwise_slim(overwrite=overwrite)

    #gaia_slim_xsdss(overwrite=overwrite)
    #gaia_clean(overwrite=overwrite)

    #G_maxs = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4]
    #G_maxs = [20.0, 20.5]
    G_maxs = [20.6]
    for G_max in G_maxs:
         merge_gaia_spzs_and_cutGmax(G_max=G_max, overwrite=overwrite)

    make_public_catalog(G_max=20.6, overwrite=overwrite)

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


def sdss_slim(overwrite=False):
    # save name
    fn_sdss_slim = '../data/SDSS_DR16Q_v4_slim.fits'
   
    # data paths 
    fn_sdss = '../data/SDSS_DR16Q_v4.fits'

    # Load data
    print("Loading data")
    tab_sdss = utils.load_table(fn_sdss)

    # Create and save
    columns_to_keep = ['SDSS_NAME', 'OBJID', 'THING_ID', 'RA', 'DEC', 'Z', 'ZWARNING', 'PSFMAG', 'PSFMAGERR']
    tab_sdss.keep_columns(columns_to_keep)

    psfmag_names = ['u', 'g', 'r', 'i', 'z']
    for i, pn in enumerate(psfmag_names):
        tab_sdss[f'{pn}_mag_sdss'] = tab_sdss['PSFMAG'][:,i]
        tab_sdss[f'{pn}_mag_err_sdss'] = tab_sdss['PSFMAGERR'][:,i]

    tab_sdss.remove_column('PSFMAG')
    tab_sdss.remove_column('PSFMAGERR')
    print(tab_sdss.columns)

    tab_sdss.write(fn_sdss_slim, overwrite=overwrite)
    print(f"Wrote table with {len(tab_sdss)} objects to {fn_sdss_slim}")


def merge_gaia_spzs_and_cutGmax(fn_spz='../data/redshift_estimates/redshifts_spz_kNN_K27_std.fits',
                                G_max=20.5, overwrite=False):

    # save name
    fn_gcat = f'../data/catalog_G{G_max}.fits'

    # data paths
    fn_gaia = '../data/gaia_candidates_clean.fits'

    # load data, cut to G_max
    tab_gaia = utils.load_table(fn_gaia)
    tab_gaia = tab_gaia[tab_gaia['phot_g_mean_mag'] < G_max]

    # SPZ-only table
    tab_spz = utils.load_table(fn_spz)
    tab_spz.keep_columns(['source_id', 'redshift_spz', 'redshift_spz_raw', 'redshift_spz_err'])

    tab_gcat = join(tab_gaia, tab_spz, keys='source_id', join_type='inner')
    utils.add_randints_column(tab_gcat)
    tab_gcat.write(fn_gcat, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gcat)} objects to {fn_gcat}")



def make_public_catalog(G_max=20.5, overwrite=False):

    # working catalog
    fn_gcat = f'../data/catalog_G{G_max}.fits'
    # update to final name choice!
    fn_public = f'../data/QUaia_G{G_max}.fits'

    tab_gcat = utils.load_table(fn_gcat)

    columns_to_keep = ['source_id', 'unwise_objid', 
                       'redshift_spz', 'redshift_spz_err', 
                       'ra', 'dec', 'l', 'b', 
                       'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                       'mag_w1_vg', 'mag_w2_vg', 
                       'pm', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error']

    # tab_public = Table(names=columns_to_keep, units=utils.label2unit_dict,
    #             descriptions=utils.label2description_dict)
    tab_public = Table()
    tab_public.meta = {'name': 'Quasars with unWISE and \emph{{Gaia}} Catalog',
                       'abbrv': 'QUaia'
                       }
    for cn in columns_to_keep:
        tab_public[cn] = tab_gcat[cn]
        tab_public[cn].info.unit = utils.label2unit_dict[cn]
        tab_public[cn].info.description = utils.label2description_dict[cn]
    
    # for tc in tab_public.columns:
    #     print(tc, tab_public[tc].info.unit)
    #print(tab_public)
    tab_public.write(fn_public, overwrite=overwrite)
    print(f"Wrote table with {len(tab_public)} objects to {fn_public}")



def gaia_candidates_plus_info(overwrite=False):

    # save to:
    fn_gaia_plus = '../data/gaia_candidates_plus.fits.gz'

    fn_gaia = '../data/gaia_candidates.fits.gz'
    fn_xwise = '../data/gaia_candidates_xunwise_all.csv'

    tab_gaia = utils.load_table(fn_gaia)
    tab_xwise = utils.load_table(fn_xwise, format='csv')
    print(f"Gaia candidates: N={len(tab_gaia)}")

    utils.add_ebv(tab_gaia)
    Rv = 3.1
    Av = Rv*tab_gaia['ebv']
    tab_gaia.add_column(Av, name='A_v')

    pm = np.sqrt(tab_gaia['pmra']**2 + tab_gaia['pmdec']**2)
    tab_gaia.add_column(pm, name='pm')

    # Cross match with unwise
    tab_xwise.keep_columns(['t1_source_id', 'mag_w1_vg', 'mag_w2_vg', 'unwise_objid'])
    tab_gaia = join(tab_gaia, tab_xwise, keys_left='source_id',            
                    keys_right='t1_source_id', join_type='left')
    tab_gaia.remove_column('t1_source_id')      

    utils.add_randints_column(tab_gaia)

    print(tab_gaia.columns)
    tab_gaia.write(fn_gaia_plus, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gaia)} objects to {fn_gaia_plus}")



def gaia_candidates_superset(overwrite=False):

    # good will have wise in it but contain all rows.
    fn_gsup = '../data/gaia_candidates_superset.fits'

    # data paths 
    fn_gaia = '../data/gaia_candidates_plus.fits.gz'
    fn_xwise = '../data/gaia_candidates_xunwise_all.csv'

    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print(f"Original Gaia table: N={len(tab_gaia)}")               

    # Require finite photometry, redshift_qsoc, and makes G cut
    tab_gsup = utils.make_superset_cuts(tab_gaia)

    # Compute the color differences
    utils.add_gaia_wise_colors(tab_gsup)

    utils.add_randints_column(tab_gsup)
    print('Final superset columns:', tab_gsup.columns)

    tab_gsup.write(fn_gsup, overwrite=overwrite)
    print(f"Wrote table with {len(tab_gsup)} objects to {fn_gsup}")


# MOVED TO DECONTAMINATE
def gaia_candidates_clean(overwrite=False):
    
    fn_gaia_clean = '../data/gaia_candidates_clean.fits'

    fn_gaia = '../data/gaia_candidates_superset.fits'
    # Load data
    print("Loading data")
    tab_gaia = utils.load_table(fn_gaia)
    print('N_gaia:', len(tab_gaia))

    fn_model = f'../data/decontamination_models/model_2lines_straight_lambda0.1.npy'
    color_cuts = np.loadtxt(fn_model)

    print("Making color cuts")
    g_w1 = tab_gaia['phot_g_mean_mag'] - tab_gaia['mag_w1_vg']
    w1_w2 = tab_gaia['mag_w1_vg'] - tab_gaia['mag_w2_vg']
    idx_clean = utils.gw1_w1w2_cuts_index(g_w1, w1_w2, color_cuts) 
    tab_gaia_clean = tab_gaia[idx_clean]

    rng = np.random.default_rng(seed=42)
    tab_gaia_clean['rand_ints'] = rng.choice(range(len(tab_gaia_clean)), size=len(tab_gaia_clean), replace=False)

    print("N_clean:", len(tab_gaia_clean))
    tab_gaia_clean.write(fn_gaia_clean, overwrite=overwrite)


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


def quasars_sdss_xgaiaall_good(overwrite=False):

    fn_sdss_xgaia_good = '../data/quasars_sdss_xgaiaall_xunwiseall_good.fits'

    print("Load in SDSS xgaia data")
    fn_sdss_xgaia = '../data/quasars_sdss_xgaiaall_sdssphot_xunwiseall.csv'
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

    # Clean out super low redshift SDSS objects, and ones with bad redshifts
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'z_sdss'
    idx_zgood = utils.redshift_cut_index(tab_sdss_xgaia, z_min, redshift_key)
    print(f"Removing {np.sum(~idx_zgood)} sources with z<{z_min}")
    tab_sdss_xgaia = tab_sdss_xgaia[idx_zgood]
    print(f"Number of SDSS QSOs with good redshfits: {len(tab_sdss_xgaia)}")
    # Note that we already did zwarning cut in Gaia cross-match, so don't need to here (didn't save zwarning)

    # get the SDSS photometry that we dropped in the initial gaia cross-match 
    # fn_sdss_full = '../data/SDSS_DR16Q_v4.fits'
    # tab_sdss_full = Table.read(fn_sdss_full, format='fits')
    # tab_sdss_full.keep_columns(['OBJID', 'PSFMAG'])
    # tab_sdss_full.rename_column('OBJID', 'objid')
    # print(len(tab_sdss_xgaia))
    # print(np.sum(np.isfinite(tab_sdss_xgaia['objid'])))
    # tab_sdss_xgaia = join(tab_sdss_xgaia, tab_sdss_full, keys='objid', join_type='left')
    # print(np.sum(tab_sdss_xgaia['PSFMAG']))
    print(tab_sdss_xgaia.columns)

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


def mcs_xgaia(overwrite=False):
    fn_mcs = '../data/mcs_xgaia.fits'

    fn_gaia = '../data/gaia_candidates_plus.fits.gz'
    #fn_gaia = '../data/gaia_candidates_superset.fits'
    tab_gaia = utils.load_table(fn_gaia)

    coord_lmc = SkyCoord('5h23m34.5s', '-69d45m22s', frame='icrs')
    coord_smc = SkyCoord('0h52m44.8s', '-72d49m43s', frame='icrs')

    sep_max_lmc = 3*u.deg
    sep_max_smc = 1.5*u.deg

    catalog = SkyCoord(ra=tab_gaia['ra'], dec=tab_gaia['dec'])

    #idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(coord_lmc, sep_max_lmc)
    seps = catalog.separation(coord_lmc)
    i_lmc = seps < sep_max_lmc

    seps = catalog.separation(coord_smc)
    i_smc = seps < sep_max_smc

    print(f'LMC: N={np.sum(i_lmc)}')
    print(f'SMC: N={np.sum(i_smc)}')
    
    tab_mcs = tab_gaia[i_lmc | i_smc]
    print(len(tab_mcs))
    tab_mcs.write(fn_mcs, overwrite=overwrite)
    print(f"Wrote MCs tab as {fn_mcs} (N={len(tab_mcs)})")


def remove_duplicate_sources(overwrite=False):

    print("Loading tables")
    fn_quasars = '../data/quasars_sdss_xgaia_xunwise_good.fits'
    fn_galaxies = '../data/galaxies_sdss_xgaia_xunwise_good.fits'
    fn_stars = '../data/stars_sdss_xgaia_xunwise_good.fits'
    # don't need to include MCs here bc there is no overlap bw MCs and SDSS

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

    # Remove duplicates from 3 SDSS tables
    for i in range(len(tabs)):
        i_dup = np.isin(tabs[i]['source_id'], source_ids_duplicate)
        print(f"Removing {np.sum(i_dup)} from {fns[i]}")
        print(f"Old table size: {len(tabs[i])}")
        tabs[i] = tabs[i][~i_dup]
        print(f"New table size: {len(tabs[i])}")

        fn_save = fns[i].split('.fits')[0] + '_nodup.fits'
        print(f"Saving to {fn_save}")
        tabs[i].write(fn_save, overwrite=overwrite)        



def make_labeled_table(overwrite=False):

    # save to:
    fn_labeled = '../data/labeled_superset.fits'

    # Requiring labeled data to be in set we will apply to, this wnec one:
    fn_gsup = '../data/gaia_candidates_superset.fits'
    tab_gsup = utils.load_table(fn_gsup)

    # Our labels come from SDSS
    tab_squasars = utils.load_table(f'../data/quasars_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS quasars: {len(tab_squasars)}")

    tab_sstars = utils.load_table(f'../data/stars_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS stars: {len(tab_sstars)}")

    tab_sgals = utils.load_table(f'../data/galaxies_sdss_xgaia_xunwise_good_nodup.fits')
    print(f"Number of SDSS galaxies: {len(tab_sgals)}")

    tab_mcs = utils.load_table(f'../data/mcs_xgaia.fits')
    print(f"Number of MCs: {len(tab_mcs)}")

    ## Stack into single table & arrays
    tab_squasars['class'] = 'q'
    tab_sstars['class'] = 's'
    tab_sgals['class'] = 'g'
    tab_mcs['class'] = 'm'

    cols_tokeep = ['source_id', 'class']

    tab_squasars.keep_columns(cols_tokeep)
    tab_sstars.keep_columns(cols_tokeep)
    tab_sgals.keep_columns(cols_tokeep)
    tab_mcs.keep_columns(cols_tokeep)

    tab_labeled = vstack([tab_squasars, tab_sstars, tab_sgals, tab_mcs], metadata_conflicts='silent')

    # Only keep labeled data that are also in our superset
    # Now that I'm only using the labeled data in wnec, i didn't need to do separate xgaia and xwise 
    # cross-matches :/ could have just crossmatched SDSS data to QSO sample. 
    # (Still useful for plotting so it's ok!)
    # We matched the SDSS samples on their gaia match's RA and dec, so the wise properties 
    # are guaranteed to be the same as the gaia candidates 
    tab_labeled_sup = join(tab_labeled, tab_gsup, join_type='inner', keys='source_id')
    print(f"N={len(tab_labeled_sup)} labeled sources are in superset (out of {len(tab_labeled)}); keeping those only")

    utils.add_randints_column(tab_labeled_sup)

    tab_labeled_sup.write(fn_labeled, overwrite=overwrite)


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


def make_labeled_sdssfootprint_table():
    fn_xsdssfootprint = '../data/gaia_candidates_xsdssfootprint.fits'
    tab_gcand_xsdssfootprint = utils.load_table(fn_xsdssfootprint)
    print(f"Number of Gaia quasar candidates in SDSS footprint: {len(tab_gcand_xsdssfootprint)}")



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



def save_as_csv(tab, column_names, fn_csv, overwrite=False):
    tab = tab.copy()
    tab.keep_columns(column_names)
    tab.write(fn_csv, format='csv', overwrite=overwrite)  
    print(f"Saved table as CSV to {fn_csv}!")



if __name__=='__main__':
    main()
