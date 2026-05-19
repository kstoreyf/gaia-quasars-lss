import numpy as np

from astropy import units as u
from astropy.table import Table, join
from astropy.coordinates import SkyCoord

import utils


def main():
    overwrite = False

    ### Make catalogs with G-cut and redshifts
    # tag_qspec = ''
    # #tag_cat = '_mags-0.05'
    # tag_cat = ''
    # G_maxs = [20.0, 20.5, 20.6]
    # #G_maxs = [20.6]
    # for G_max in G_maxs:
    #     merge_gaia_spzs_and_cutGmax(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)

    # ### Make public-facing catalog
    #tag_qspec = ''
    #tag_cat = '_mags-0.05'
    #tag_cat = ''
    #G_maxs = [20.0, 20.5]
    #for G_max in G_maxs:
    #    make_public_catalog(G_max=G_max, tag_qspec=tag_qspec, tag_cat=tag_cat, overwrite=overwrite)

    # ### Make redshift-split catalogs
    #G_max = 20.5
    #n_zbins = 4
    #make_redshift_split_catalogs(G_max, n_zbins)

    ### Make redshift-split catalogs for CIB analysis by Giulia
    # G_max = 20.5
    # z_bins = [0, 1.0, 2.3, 5]
    # make_redshift_split_catalogs(G_max, z_bins=z_bins, save_tag='CIB')
    # z_bins = [0,  0.5, 1.0, 1.5, 2.0, 2.5, 5]
    # make_redshift_split_catalogs(G_max, z_bins=z_bins, save_tag='CIB')

    ### Make redshift-split catalogs for autocorr-dutycycle analysis by christina eilers & mariona
    #G_max = 20.5
    #z_bins = [0.0,1.0,2.0,3.0,4.0]
    #z_bins = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    #make_redshift_split_catalogs(G_max, z_bins=z_bins)
    #z_bins = [2.9,3.5,5.0]
    #make_redshift_split_catalogs(G_max, z_bins=z_bins)
    #z_bins = [0.8,1.2] #Shen+07 bin, most data
    #make_redshift_split_catalogs(G_max, z_bins=z_bins)

    ### For Paul quaia-desi comparison
    G_max = 20.5
    z_bins = [0.8,2.1] #desi qso range
    make_redshift_split_catalogs(G_max, z_bins=z_bins)

    #crossmatch_quaia_sdss_dr16q_prop()

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


def _to_deg(col):
    """Convert column to plain degree array (handles Quantity, masked, etc.)."""
    data = col.data if hasattr(col, 'mask') else col
    a = np.asarray(data)
    if hasattr(a, 'to'):
        a = a.to(u.deg).value
    elif hasattr(a, 'value'):
        a = np.asarray(a.value)
    a = np.asarray(a, dtype=float)
    if hasattr(col, 'mask') and np.any(col.mask):
        a = np.where(col.mask, np.nan, a)
    return a


def crossmatch_quaia_sdss_dr16q_prop(G_max=20.5, match_radius_arcsec=1.0, overwrite=False):
    """
    Cross-match QUaIA G{G_max} with SDSS DR16Q (sky position), keep matches within
    match_radius_arcsec. Save an astropy table with columns: source_id (QUaIA),
    sdss_objid (SDSS OBJID from DR16Q).
    """
    fn_quaia = f'../data/quaia_G{G_max}.fits'
    fn_dr16q = '../data/dr16q_prop_May01_2024.fits.gz'
    fn_out = f'../data/match_quaia_G{G_max}_sdss_dr16q_prop.fits'

    tab_quaia = utils.load_table(fn_quaia)
    tab_dr16q = Table.read(fn_dr16q, hdu=1)

    ra_quaia = _to_deg(tab_quaia['ra'])
    dec_quaia = _to_deg(tab_quaia['dec'])
    ra_dr16 = _to_deg(tab_dr16q['RA'] if 'RA' in tab_dr16q.colnames else tab_dr16q['PLUG_RA'])
    dec_dr16 = _to_deg(tab_dr16q['DEC'] if 'DEC' in tab_dr16q.colnames else tab_dr16q['PLUG_DEC'])

    coords_quaia = SkyCoord(ra=ra_quaia * u.deg, dec=dec_quaia * u.deg, frame='icrs')
    coords_dr16q = SkyCoord(ra=ra_dr16 * u.deg, dec=dec_dr16 * u.deg, frame='icrs')

    idx_quaia, sep2d, _ = coords_dr16q.match_to_catalog_sky(coords_quaia)
    within = sep2d < match_radius_arcsec * u.arcsec

    quaia_source_id = tab_quaia['source_id'][idx_quaia[within]]
    sdss_objid = tab_dr16q['OBJID'][within]

    tab_match = Table()
    tab_match['source_id'] = quaia_source_id
    tab_match['sdss_objid'] = sdss_objid
    tab_match.meta['description'] = f'QUaIA G{G_max} × SDSS DR16Q, sep < {match_radius_arcsec} arcsec'

    tab_match.write(fn_out, overwrite=overwrite)
    print(f"Quaia G{G_max} × SDSS DR16Q: {np.sum(within)} matches within {match_radius_arcsec} arcsec -> {fn_out}")


def make_redshift_split_catalogs(G_max, n_zbins=None, z_bins=None, overwrite=True,
                                 save_tag=''):

    assert n_zbins is not None or z_bins is not None, "Either n_zbins or z_bins must be passed!"

    if z_bins is not None and n_zbins is not None:
        print("z_bins passed, ignoring n_zbins")
    if z_bins is not None:
        n_zbins = len(z_bins)-1
        print(f"z_bins: {z_bins}, setting n_zbins={n_zbins}")

    fn_gcat = f'../data/quaia_G{G_max}.fits'
    tab_gcat = utils.load_table(fn_gcat)

    if z_bins is None:
        z_percentiles = np.linspace(0.0, 100.0, n_zbins+1)
        print(z_percentiles)
        z_bins = np.percentile(list(tab_gcat['redshift_quaia']), z_percentiles)
        z_bins[-1] += 0.01 # add a bit to maximum bin to make sure the highest-z source gets included
        z_bins[0] -= 0.01 # add a bit to minimum bin to make sure the lowest-z source gets included

    print("zbins:", z_bins)
    print("n_zbins:", n_zbins)

    for bb in range(n_zbins):
        i_zbin = (tab_gcat['redshift_quaia'] >= z_bins[bb]) & (tab_gcat['redshift_quaia'] < z_bins[bb+1])
        tab_gcat_zbin = tab_gcat[i_zbin]
        if z_bins is None:
            fn_gcat_zbin = f'../data/quaia_G{G_max}_zsplit{n_zbins}bin{bb}{save_tag}.fits'
        else:
            fn_gcat_zbin = f'../data/quaia_G{G_max}_zmin{z_bins[bb]}zmax{z_bins[bb+1]}{save_tag}.fits' 
        tab_gcat_zbin.write(fn_gcat_zbin, overwrite=overwrite)
        print("zmin:", np.min(tab_gcat_zbin['redshift_quaia']))
        print("zmax:", np.max(tab_gcat_zbin['redshift_quaia']))
        print(f"Wrote table with {len(tab_gcat_zbin)} objects to {fn_gcat_zbin}")


if __name__=='__main__':
    main()
