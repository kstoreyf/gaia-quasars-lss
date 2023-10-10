import numpy as np

from astropy.table import Table, join

import utils


def main():
    overwrite = True

    ### Make catalogs with G-cut and redshifts
    # tag_qspec = ''
    # #tag_cat = '_mags-0.05'
    # tag_cat = ''
    # G_maxs = [20.0, 20.5, 20.6]
    # #G_maxs = [20.6]
    # for G_max in G_maxs:
    #     merge_gaia_spzs_and_cutGmax(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)

    # ### Make public-facing catalog
    # tag_qspec = ''
    # #tag_cat = '_mags-0.05'
    # tag_cat = ''
    # G_maxs = [20.0, 20.5]
    # for G_max in G_maxs:
    #     make_public_catalog(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)

    ### Make redshift-split catalogs
    G_max = 20.5
    n_zbins = 2
    make_redshift_split_catalogs(G_max, n_zbins)


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
    fn_public = f'../data/quaia_G{G_max}{tag_qspec}{tag_cat}.fits'

    tab_gcat = utils.load_table(fn_gcat)

    columns_to_keep = ['source_id', 'unwise_objid', 
                       'redshift_spz', 'redshift_spz_err', 
                       'ra', 'dec', 'l', 'b', 
                       'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                       'mag_w1_vg', 'mag_w2_vg', 
                       'pm', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error']

    tab_public = Table()
    tab_public.meta = {'name': '\emph{{Gaia}}--\emph{{unWISE}} Quasar Catalog',
                       'abbrv': 'Quaia'
                       }

    rename_dict = {'redshift_spz': 'redshift_quaia',
                   'redshift_spz_err': 'redshift_quaia_err'
                   }

    for cn in columns_to_keep:
        if cn in rename_dict:
            cn_new = rename_dict[cn]
        else: 
            cn_new = cn
        tab_public[cn_new] = tab_gcat[cn]
        tab_public[cn_new].info.unit = utils.label2unit_dict[cn_new]
        tab_public[cn_new].info.description = utils.label2description_dict[cn_new]
    
    # for tc in tab_public.columns:
    #     print(tc, tab_public[tc].info.unit)
    print(tab_public.columns)
    tab_public.write(fn_public, overwrite=overwrite)
    print(f"Wrote table with {len(tab_public)} objects to {fn_public}")


def make_redshift_split_catalogs(G_max, n_zbins, overwrite=True):

    fn_gcat = f'../data/quaia_G{G_max}.fits'
    tab_gcat = utils.load_table(fn_gcat)
    z_percentiles = np.linspace(0.0, 100.0, n_zbins+1)
    print(z_percentiles)
    z_bins = np.percentile(list(tab_gcat['redshift_quaia']), z_percentiles)
    z_bins[-1] += 0.01 # add a bit to maximum bin to make sure the highest-z source gets included
    z_bins[0] -= 0.01 # add a bit to minimum bin to make sure the lowest-z source gets included
    print("zbins:", z_bins)

    for bb in range(n_zbins):
        i_zbin = (tab_gcat['redshift_quaia'] >= z_bins[bb]) & (tab_gcat['redshift_quaia'] < z_bins[bb+1])
        tab_gcat_zbin = tab_gcat[i_zbin]
        fn_gcat_zbin = f'../data/quaia_G{G_max}_zsplit{n_zbins}bin{bb}.fits'
        tab_gcat_zbin.write(fn_gcat_zbin, overwrite=overwrite)
        print("zmin:", np.min(tab_gcat_zbin['redshift_quaia']))
        print("zmax:", np.max(tab_gcat_zbin['redshift_quaia']))
        print(f"Wrote table with {len(tab_gcat_zbin)} objects to {fn_gcat_zbin}")


if __name__=='__main__':
    main()
