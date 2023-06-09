import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack

import utils

"""
make_data_tables.py

- To make superset:
    - gaia_candidates_plus_info()
        - in: gaia_candidates.fits.gz
        - out: gaia_candidates_plus.fits
    - gaia_candidates_superset()
        - in: gaia_candidates_plus.fits, gaia_candidates_xunwise_all.csv
        - out: gaia_candidates_superset.fits
- To make labeled data for training / validation (<obj_type> = [quasars, stars, galaxies]):
    - <obj_type>_sdss_xgaia_good()
        - in: <obj_type>_sdss_xgaia_xunwise_all.csv
        - out: <obj_type>_sdss_xgaia_xunwise_good.fits
    - remove_duplicate_sources()
        - in: <obj_type>_sdss_xgaia_xunwise_good.fits (all 3)
        - out: <obj_type>_sdss_xgaia_xunwise_good_nodup.fits (all 3)
    - make_labeled_table()
        - in: <obj_type>_sdss_xgaia_xunwise_good_nodup.fits (all 3), gaia_candidates_superset.fits
        - out: labeled_xsuperset.fits
"""

def main():
    overwrite = True

    ### Make main datasets
    #gaia_candidates_plus_info(overwrite=overwrite)
    #gaia_candidates_superset(overwrite=overwrite)
    ###gaia_candidates_clean(overwrite=overwrite)

    #gaia_slim(overwrite=overwrite)
    #sdss_slim(overwrite=overwrite)
    #eboss_slim(overwrite=overwrite)
    #gaia_purer_sourceids(overwrite=overwrite)

    ### Make quasar tables
    #quasars_sdss_xgaia_good(overwrite=overwrite)
    #quasars_sdss_xgaiaall_good(overwrite=overwrite)
    #quasars_eboss_xgaia_good(overwrite=overwrite)
    #quasars_eboss_xgaiaall_good(overwrite=overwrite)

    ### Make other labeled tables
    # galaxies_sdss_xgaia_good(overwrite=overwrite)
    # stars_sdss_xgaia_good(overwrite=overwrite)
    #mcs_xgaia(overwrite=overwrite)

    ### Adjust and combine labeled data
    #remove_duplicate_sources(overwrite=overwrite)
    #make_labeled_table(overwrite=overwrite)
    
    #get_gaia_xsdssfootprint(overwrite=overwrite)

    #gaia_unwise_slim(overwrite=overwrite)
    #gaia_catwise_slim(overwrite=overwrite)

    #gaia_slim_xsdss(overwrite=overwrite)
    #gaia_clean(overwrite=overwrite)

    #perturbed_magnitude_catalogs(mag_perturb=-0.05, overwrite=overwrite)

    tag_qspec = ''
    tag_cat = '_mags-0.05'
    G_maxs = [20.0, 20.5]
    #G_maxs = [20.6]
    for G_max in G_maxs:
        merge_gaia_spzs_and_cutGmax(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)
        make_public_catalog(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)

    # G_max = 20.5
    # n_zbins = 3
    # make_redshift_split_catalogs(G_max, n_zbins)

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



def eboss_slim(overwrite=False):
    # save name
    fn_eboss_slim = '../data/eboss_quasars_slim.fits'
   
    # data paths 
    fn_eboss = '../data/eBOSS_QSO_full_ALLdata-vDR16_changecolname.fits'

    # Load data
    print("Loading data")
    tab_eboss = utils.load_table(fn_eboss)

    print(tab_eboss['Z'])

    # Create and save
    columns_to_keep = ['PLATE', 'MJD', 'FIBERID', 'ID', 'QSO_ID', 'RA', 'DEC', 'Z', 'ZWARNING', 
                       'MODELMAG', 'IMATCH', 'COMP_BOSS', 'sector_SSR']
    tab_eboss.keep_columns(columns_to_keep)

    mag_names = ['u', 'g', 'r', 'i', 'z']
    for i, mn in enumerate(mag_names):
        tab_eboss[f'{mn}_mag_sdss'] = tab_eboss['MODELMAG'][:,i]

    tab_eboss.remove_column('MODELMAG')
    print(tab_eboss.columns)

    print(tab_eboss['Z'])

    tab_eboss.write(fn_eboss_slim, overwrite=overwrite)
    print(f"Wrote table with {len(tab_eboss)} objects to {fn_eboss_slim}")



def merge_gaia_spzs_and_cutGmax(G_max=20.5, tag_qspec='', tag_cat='', overwrite=False):

    # save name
    fn_gcat = f'../data/catalog_G{G_max}{tag_qspec}{tag_cat}.fits'

    # data paths
    fn_gaia = f'../data/gaia_candidates_clean{tag_qspec}{tag_cat}.fits'
    fn_spz = f'../data/redshift_estimates/redshifts_spz{tag_qspec}{tag_cat}_kNN_K27_std.fits'

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



def make_public_catalog(G_max=20.5, tag_qspec='', tag_cat='', overwrite=False):

    # working catalog
    fn_gcat = f'../data/catalog_G{G_max}{tag_qspec}{tag_cat}.fits'
    # update to final name choice!
    fn_public = f'../data/QUaia_G{G_max}{tag_qspec}{tag_cat}.fits'

    tab_gcat = utils.load_table(fn_gcat)

    columns_to_keep = ['source_id', 'unwise_objid', 
                       'redshift_spz', 'redshift_spz_err', 
                       'ra', 'dec', 'l', 'b', 
                       'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                       'mag_w1_vg', 'mag_w2_vg', 
                       'pm', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error']

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


def make_redshift_split_catalogs(G_max, n_zbins, overwrite=True):

    fn_gcat = f'../data/QUaia_G{G_max}.fits'
    tab_gcat = utils.load_table(fn_gcat)
    z_percentiles = np.linspace(0.0, 100.0, n_zbins+1)
    print(z_percentiles)
    z_bins = np.percentile(list(tab_gcat['redshift_spz']), z_percentiles)
    z_bins[-1] += 0.01 # add a bit to maximum bin to make sure the highest-z source gets included
    print("zbins:", z_bins)

    for bb in range(n_zbins):
        i_zbin = (tab_gcat['redshift_spz'] >= z_bins[bb]) & (tab_gcat['redshift_spz'] < z_bins[bb+1])
        tab_gcat_zbin = tab_gcat[i_zbin]
        fn_gcat_zbin = f'../data/QUaia_G{G_max}_zsplit{n_zbins}bin{bb}.fits'
        tab_gcat_zbin.write(fn_gcat_zbin, overwrite=overwrite)
        print("zmin:", np.min(tab_gcat_zbin['redshift_spz']))
        print("zmax:", np.max(tab_gcat_zbin['redshift_spz']))
        print(f"Wrote table with {len(tab_gcat_zbin)} objects to {fn_gcat_zbin}")




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



def quasars_sdss_xgaia_good(overwrite=False):

    fn_sdss_xgaia_good = '../data/quasars_sdss_xgaia_xunwise_good.fits'

    print("Load in SDSS xgaia data")
    fn_sdss_xgaia = '../data/quasars_sdss_xgaia_xunwise_all.csv'
    tab_sdss_xgaia = utils.load_table(fn_sdss_xgaia, format='csv')
    print(f"Number of SDSS xGaia QSOs: {len(tab_sdss_xgaia)}")
    update_column_names_sdss(tab_sdss_xgaia)

    # This cut also serves to keep only sources with Gaia data!! 
    # Keep sources with sufficient phot_bp_n_obs and phot_rp_n_obs
    i_good_nobs = cuts_good_nobs(tab_sdss_xgaia)
    tab_sdss_xgaia = tab_sdss_xgaia[i_good_nobs]

    i_cuts_sdss = cuts_quasars_sdss(tab_sdss_xgaia)
    tab_sdss_xgaia = tab_sdss_xgaia[i_cuts_sdss]

    tab_sdss_xgaia.write(fn_sdss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_sdss_xgaia)} objects to {fn_sdss_xgaia_good}")


def quasars_sdss_xgaiaall_good(overwrite=False):

    fn_sdss_xgaia_good = '../data/quasars_sdss_xgaiaall_xunwiseall_good.fits'

    print("Load in SDSS xgaia data")
    fn_sdss_xgaia = '../data/quasars_sdss_xgaiaall_sdssphot_xunwiseall.csv'
    tab_sdss_xgaia = utils.load_table(fn_sdss_xgaia, format='csv')
    print(f"Number of SDSS xGaia QSOs: {len(tab_sdss_xgaia)}")

    update_column_names_sdss(tab_sdss_xgaia)

    i_cuts_sdss = cuts_quasars_sdss(tab_sdss_xgaia)
    tab_sdss_xgaia = tab_sdss_xgaia[i_cuts_sdss]

    print(tab_sdss_xgaia.columns)

    tab_sdss_xgaia.write(fn_sdss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_sdss_xgaia)} objects to {fn_sdss_xgaia_good}")


def quasars_eboss_xgaia_good(overwrite=False):

    fn_eboss_xgaia_good = '../data/quasars_eboss_xgaia_xunwise_good.fits'

    print("Load in eBOSS xgaia data")
    fn_eboss_xgaia = '../data/quasars_eboss_xgaiaall_xunwiseall.csv'
    tab_eboss_xgaia = utils.load_table(fn_eboss_xgaia, format='csv')
    print(f"Number of eBOSS xGaia QSOs: {len(tab_eboss_xgaia)}")

    update_column_names_sdss(tab_eboss_xgaia)

    # Keep sources with sufficient phot_bp_n_obs and phot_rp_n_obs
    i_good_nobs = cuts_good_nobs(tab_eboss_xgaia)
    tab_eboss_xgaia = tab_eboss_xgaia[i_good_nobs]

    # apply cuts
    i_eboss_cuts = cuts_quasars_eboss(tab_eboss_xgaia)
    tab_eboss_good = tab_eboss_xgaia[i_eboss_cuts]

    tab_eboss_good.write(fn_eboss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_eboss_good)} objects to {fn_eboss_xgaia_good}")


def quasars_eboss_xgaiaall_good(overwrite=False):

    fn_eboss_xgaia_good = '../data/quasars_eboss_xgaiaall_xunwiseall_good.fits'

    print("Load in eBOSS xgaia data")
    fn_eboss_xgaia = '../data/quasars_eboss_xgaiaall_xunwiseall.csv'
    tab_eboss_xgaia = utils.load_table(fn_eboss_xgaia, format='csv')
    print(f"Number of eBOSS xGaia QSOs: {len(tab_eboss_xgaia)}")

    update_column_names_sdss(tab_eboss_xgaia)

    # apply cuts
    i_eboss_cuts = cuts_quasars_eboss(tab_eboss_xgaia)
    tab_eboss_good = tab_eboss_xgaia[i_eboss_cuts]

    tab_eboss_good.write(fn_eboss_xgaia_good, overwrite=overwrite)
    print(f"Wrote table with {len(tab_eboss_good)} objects to {fn_eboss_xgaia_good}")


def cuts_good_nobs(tab):
    i_good_nobs = (tab['phot_bp_n_obs'] >= 5) & (tab['phot_rp_n_obs'] >= 5)
    print(f"N={np.sum(~i_good_nobs)} sources with <5 bp or rp n_obs")
    return i_good_nobs


def update_column_names_sdss(tab):
    tab.rename_column('ra', 'ra_unwise')
    tab.rename_column('dec', 'dec_unwise')
    for column_name in list(tab.columns):
        new_name = column_name
        if column_name.startswith('t1'):
            new_name = column_name.split('t1_')[-1]
            # don't need this right now bc changed name in sql script,
            # but keep in case
            if new_name=='z':
                new_name = 'z_sdss'
        tab.rename_column(column_name, new_name)    


def cuts_quasars_sdss(tab_sdss):
    # NOTE that we already did zwarning cut in Gaia cross-match, 
    # so don't need to here (didn't save zwarning)

    
    # Clean out super low redshift SDSS objects, and ones with bad redshifts
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'z_sdss'
    i_zgood = utils.redshift_cut_index(tab_sdss, z_min, redshift_key)
    print(f"{np.sum(~i_zgood)} sources with z<{z_min}")
    return i_zgood


def cuts_quasars_eboss(tab_eboss):
    # Clean out super low redshift SDSS objects, and ones with bad redshifts
    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'z_sdss'
    i_zgood = utils.redshift_cut_index(tab_eboss, z_min, redshift_key)
    print(f"{np.sum(~i_zgood)} sources with z<{z_min}")
    
    ### eBOSS data choices
    # eBOSS and legacy quasars (as used in clustering sample)
    i_imatch = (tab_eboss['imatch']==1) | (tab_eboss['imatch']==2)
    print('N imatch:', np.sum(i_imatch))
    # >0.5 sector completeness and redshift goodness (as used in clustering sample)
    i_comp = (tab_eboss['comp_boss']>0.5)
    i_sect = (tab_eboss['sector_ssr']>0.5)
    print('N comp_boss:', np.sum(i_comp), 'N sector_ssr:', np.sum(i_sect))
    print('N comp_boss & sector_ssr:', np.sum(i_comp & i_sect))
    i_clust = i_imatch & i_comp & i_sect
    print('N clust:', np.sum(i_clust))
    # zwarning must == 0
    i_zwarning0 = tab_eboss['zwarning']==0

    i_eboss = i_clust & i_zgood & i_zwarning0
    return i_eboss



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
    name_qspec = 'eboss'
    tag_qspec = '_qeboss'
    fn_quasars = f'../data/quasars_{name_qspec}_xgaia_xunwise_good.fits'
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

        fn_save = fns[i].split('.fits')[0] + f'_nodup{tag_qspec}.fits'
        print(f"Saving to {fn_save}")
        tabs[i].write(fn_save, overwrite=overwrite)        



def make_labeled_table(overwrite=False):

    # name_qspec = 'sdss'
    # tag_qspec = ''
    name_qspec = 'eboss'
    tag_qspec = '_qeboss'

    # save to:
    fn_labeled = f'../data/labeled_superset{tag_qspec}.fits'

    # Requiring labeled data to be in set we will apply to, this wnec one:
    fn_gsup = '../data/gaia_candidates_superset.fits'
    tab_gsup = utils.load_table(fn_gsup)

    # Our labels come from SDSS
    tab_squasars = utils.load_table(f'../data/quasars_{name_qspec}_xgaia_xunwise_good_nodup{tag_qspec}.fits')
    print(f"Number of SDSS quasars: {len(tab_squasars)}")

    tab_sstars = utils.load_table(f'../data/stars_sdss_xgaia_xunwise_good_nodup{tag_qspec}.fits')
    print(f"Number of SDSS stars: {len(tab_sstars)}")

    tab_sgals = utils.load_table(f'../data/galaxies_sdss_xgaia_xunwise_good_nodup{tag_qspec}.fits')
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
    # NOTE except for the eboss quasars, which we matched on sdss ra and dec bc not all have gaia... 
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


def perturbed_magnitude_catalogs(mag_perturb=0.05, overwrite=False):

    tag_cat = f'_mags{mag_perturb}'

    # the only one that will matter is G in current setup, but perturb all for completeness
    mag_names = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                 'mag_w1_vg', 'mag_w2_vg', 
                 'u_mag_sdss', 'g_mag_sdss', 'r_mag_sdss', 'i_mag_sdss', 'z_mag_sdss']

    # name_qspec = 'eboss'
    # tag_qspec = '_qeboss'
    name_qspec = 'sdss'
    tag_qspec = ''

    fn_gsup = '../data/gaia_candidates_superset.fits'
    fn_labeled = f'../data/labeled_superset{tag_qspec}.fits'
    fn_sdss = f'../data/quasars_{name_qspec}_xgaia_xunwise_good_nodup{tag_qspec}.fits'
    # don't need the good_nodup of gals and stars bc those just go into fn_labeled;
    # only need fn_sdss independently (for specphotoz, and decontaminate to prep for that)
    # dont need quasars xgaiaall because only use that for plotting, comparison (i think)

    fns = [fn_gsup, fn_labeled, fn_sdss]
    for fn in fns:
        print(fn)
        tab = utils.load_table(fn)
        #print(tab['phot_g_mean_mag'])
        for mag_name in mag_names:
            if mag_name in tab.columns:
                tab[mag_name] += mag_perturb
        #print(tab['phot_g_mean_mag'])            
        fn_save = fn.split('.fits')[0] + tag_cat + '.fits'
        tab.write(fn_save, overwrite=overwrite)
        print(f"Wrote table with {len(tab)} objects to {fn_save}")


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
