import numpy as np

import healpy as hp
import pymaster as nmt

import utils
import masks


def main():
    #NSIDE = 2048
    NSIDE = 256

    G_max = 20
    fn_gaia = f'../data/gaia_G{G_max}.fits'
    fn_rand = f'../data/randoms/random_stardustm1064_G{G_max}_10x.fits'
    mask_names_gaia = ['mcs', 'dust']
    Av_max = 0.2
    fn_Cls = f'../data/Cls/Cls_G{G_max}_NSIDE{NSIDE}.npy'
    
    print(f"Computing lensing-QSO cross-correlation for QSOs with G<{G_max}, maps with NSIDE={NSIDE}")
    print(f"Will save Cls to {fn_Cls}")

    mask_lensing = get_planck_lensing_mask(NSIDE)
    map_lensing = get_planck_lensing_map(NSIDE)

    mask_overdensity = get_qso_mask(NSIDE, mask_names_gaia, Av_max=Av_max)
    map_overdensity = get_qso_overdensity_map(NSIDE, fn_gaia, fn_rand, mask_overdensity)

    # "We choose a conservative binning scheme with linearly spaced bins of
    # size ∆l = 50 starting from l_min = 25."
    ell_min = 25
    ell_max = 600
    ell_bin_width = 50
    bins = get_bins_linear(ell_min, ell_max, ell_bin_width)

    Cls_kk_obj = compute_Cls(bins, map_lensing, map_lensing, mask_lensing, mask_lensing)
    Cls_kq_obj = compute_Cls(bins, map_lensing, map_overdensity, mask_lensing, mask_overdensity)
    Cls_qq_obj = compute_Cls(bins, map_overdensity, map_overdensity, mask_overdensity, mask_overdensity)

    ell_arr = bins.get_effective_ells()
    Cl_objs = [Cls_kk_obj, Cls_kq_obj, Cls_qq_obj]
    result = np.array([ell_arr, Cls_kk_obj[0], Cls_kq_obj[0], Cls_qq_obj[0], bins, Cl_objs])
    np.save(fn_Cls, result)
    print(f"Saved Cls to {fn_Cls}")


#Data from https://pla.esac.esa.int/#cosmology, (cosmology tab then lensing tab)
#Details: https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Lensing
def get_planck_lensing_map(NSIDE, fn_lensing='../data/COM_Lensing_4096_R3.00/MV/dat_klm.fits',
                           lmax=4096, lmax_smooth=2500):
    print("Getting Planck lensing map")
    # Guidance here from: https://zonca.dev/2020/09/planck-spectra-healpy.html
    alm_lensing = hp.read_alm(fn_lensing)
    map_lensing = hp.alm2map(alm_lensing, nside=NSIDE, lmax=lmax)
    # can't figure out how to set lmin! but this fixes the scaling
    # should i mask and then smooth or vice versa??
    if lmax_smooth is not None:
        map_lensing = hp.smoothing(map_lensing, lmax=lmax_smooth)
    return map_lensing
    

def get_planck_lensing_mask(NSIDE, fn_mask='../data/COM_Lensing_4096_R3.00/mask.fits.gz',
                            aposize_deg=0.5):
    print("Getting Planck lensing mask")
    # Apodization: 0.5 deg = 30' (arcmin), used in Martin White paper
    mask_lensing = hp.read_map(fn_mask, dtype=bool)
    mask_lensing = hp.pixelfunc.ud_grade(mask_lensing, NSIDE)
    if aposize_deg is not None:
        mask_lensing = nmt.mask_apodization(mask_lensing, aposize_deg, apotype="C2")
    return mask_lensing


# TODO: maybe don't need to mask here bc we'll input the mask to the Cls?? 
# but this changes the mean, so maybe should first too, as it says in White paper?
# tho maybe cleaner way to do with healpy...
def get_qso_overdensity_map(NSIDE, fn_gaia, fn_rand, mask_qso):
    print("Getting QSO overdensity map")

    tab_gaia = utils.load_table(fn_gaia)
    tab_rand = utils.load_table(fn_rand)

    idx_keep_gaia = masks.subsample_mask_indices(NSIDE, tab_gaia['ra'], tab_gaia['dec'], mask_qso)
    tab_gaia = tab_gaia[idx_keep_gaia]
    idx_keep_rand = masks.subsample_mask_indices(NSIDE, tab_rand['ra'], tab_rand['dec'], mask_qso)
    tab_rand = tab_rand[idx_keep_rand]

    map_nqso_gaia, _ = utils.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)
    map_nqso_rand, _ = utils.get_map(NSIDE, tab_rand['ra'], tab_rand['dec'], null_val=0)

    #"The weighted random counts in each Healpix pixel then form the “random map”. The overdensity field is defined as the “LRG map” divided by the “random map”, normalized to mean density and mean subtracted." (White 2022)
    #TODO: figure out what to do about these zeros!
    map_overdensity = map_nqso_gaia / map_nqso_rand
    idx_odens_finite = np.isfinite(map_overdensity)
    map_overdensity /= np.mean(map_overdensity[idx_odens_finite]) #?? is this what normalized to mean density means?
    map_overdensity -= np.mean(map_overdensity[idx_odens_finite])
    map_overdensity[~idx_odens_finite] = hp.UNSEEN
    return map_overdensity


def get_qso_mask(NSIDE, mask_names_gaia, b_max=None, Av_max=None, R=3.1):
    print("Getting QSO mask")

    fn_dustmap = f'../data/maps/map_dust_NSIDE{NSIDE}.npy'
    # dict points to tuple with masks and extra args
    mask_gaia_dict = {'plane': (masks.galactic_plane_mask, [b_max]),
                  'mcs': (masks.magellanic_clouds_mask, []),
                  'dust': (masks.galactic_dust_mask, [Av_max, R, fn_dustmap])}
    NPIX = hp.nside2npix(NSIDE)
    # masks have 1s where to mask. if current mask OR new
    # mask has a 1, want a 1, so we need OR
    mask_qso = np.zeros(NPIX, dtype=bool) # zeros mean no mask
    for mask_name in mask_names_gaia:
        mask_func, mask_func_args = mask_gaia_dict[mask_name]
        mask = mask_func(NSIDE, *mask_func_args)
        mask_qso = (mask_qso | mask)
    return mask_qso


def get_mask_indices_keep(NSIDE, ra, dec, mask_names_gaia):
    b_max = 10
    Av_max = 0.2
    R = 3.1
    fn_dustmap = f'../data/maps/map_dust_NSIDE{NSIDE}.npy'
    # dict points to tuple with masks and extra args
    mask_gaia_dict = {'plane': (masks.galactic_plane_mask, [b_max]),
                  'mcs': (masks.magellanic_clouds_mask, []),
                  'dust': (masks.galactic_dust_mask, [Av_max, R, fn_dustmap])}

    idx_keep = np.full(len(ra),True)
    for mask_name in mask_names_gaia:
        mask_func, mask_func_args = mask_gaia_dict[mask_name]
        idx_keep_m = masks.subsample_by_mask(NSIDE, ra[idx_keep], dec[idx_keep], 
                                             mask_func, mask_func_args)
        idx_keep = idx_keep & idx_keep_m
    return idx_keep

## Compute pseudo-Cls
#Following https://arxiv.org/pdf/2111.09898.pdf and https://namaster.readthedocs.io/en/latest/sample_simple.html
def compute_Cls(bins, map1, map2, mask1, mask2):
    print("Computing Cls")
    field1 = nmt.NmtField(mask1, [map1])
    field2 = nmt.NmtField(mask2, [map2])
    Cls = nmt.compute_full_master(field1, field2, bins)
    return Cls


def get_bins_linear(ell_min, ell_max, ell_bin_width):
    ell_edges = np.arange(ell_min, ell_max+ell_bin_width, ell_bin_width)
    ell_ini = ell_edges[:-1]
    ell_end = ell_edges[1:]
    bins = nmt.NmtBin.from_edges(ell_ini, ell_end)
    return bins



if __name__=='__main__':
    main()
