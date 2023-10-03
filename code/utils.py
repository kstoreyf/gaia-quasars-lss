import numpy as np
import os
import pandas as pd

import healpy as hp
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
from dustmaps.sfd import SFDQuery
from dustmaps.csfd import CSFDQuery

import matplotlib
from matplotlib import pyplot as plt


def jackknife(func, data, rand, *args, **kwargs):
    n = 12 # magic
    l_name = "l"
    assert np.all(data[l_name] >= 0.*u.deg) and np.all(data[l_name] < 360.*u.deg) # seriously; degrees?
    assert np.all(rand[l_name] >= 0.*u.deg) and np.all(rand[l_name] < 360.*u.deg)
    dl = 360. / n
    def one_jack(i):
        l1 = (i * dl)*u.deg
        l2 = ((i + 1) * dl)*u.deg
        idx_data = (data[l_name] < l1) | (data[l_name] >= l2)
        idx_rand = (rand[l_name] < l1) | (rand[l_name] >= l2)
        kwargs['jack'] = i
        return func(data[idx_data], rand[idx_rand], *args, **kwargs)
    outs = np.array(list(map(one_jack, range(n))))
    return jackknife_mean_var(outs)


def jackknife_mean_var(values):
    values = np.array(values)
    n = values.shape[0]
    print(f"Jackknife mean & var with n={n}")
    mean = np.mean(values, axis=0)
    var = ((n - 1) / n) * np.sum((values - mean) ** 2, axis=0)
    return mean, var


def get_fraction_recovered(Y_true, Y_hat, dz):
        return np.sum(np.abs(Y_hat - Y_true)/(1 + Y_true) < dz) / len(Y_true)


def add_gaia_wise_colors(tab, w1_name='mag_w1_vg', w2_name='mag_w2_vg'):
    g = tab['phot_g_mean_mag']
    bp = tab['phot_bp_mean_mag']
    rp = tab['phot_rp_mean_mag']
    w1 = tab[w1_name]
    w2 = tab[w2_name]

    tab.add_column(g-rp, name='g_rp')
    tab.add_column(bp-g, name='bp_g')
    tab.add_column(bp-rp, name='bp_rp')
    tab.add_column(g-w1, name='g_w1')
    tab.add_column(w1-w2, name='w1_w2')


# CHANGED from tab input to g_w1, w1_w2
def gw1_w1w2_cuts_index(g_w1, w1_w2, color_cuts):
    # start with all
    idx_clean = np.full(len(g_w1), True)
    for cut in color_cuts:
        idx_colorcut = gw1_w1w2_cut_index(g_w1, w1_w2, cut)
        idx_clean = idx_clean & idx_colorcut    
    return idx_clean

def gw1_w1w2_cut_index(g_w1, w1_w2, cut):
    return cut[0]*g_w1 + cut[1]*w1_w2 > cut[2]

def cuts_index_straight(colors, cuts_min):
    # start with all
    i_clean = np.full(len(colors[0]), True)
    for color, cut in zip(colors, cuts_min):
        i_colorcut = color > cut
        i_clean = i_clean & i_colorcut    
    return i_clean
    

def cuts_index(color_arr, slopes, intercepts):
    color_arr = np.array(color_arr)
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    #assert color_arr.shape[1]==slopes.shape[1], "Should have same # colors and slopes"
    assert slopes.shape[0]==intercepts.shape[0], "Should have same # slopes and intercepts"
    i_clean = np.full(color_arr.shape[0], True)
    for ii in range(len(intercepts)):
        #print((color_arr*slopes[ii]).shape)
        i_makescut = np.dot(color_arr, slopes[ii]) > intercepts[ii]
        #i_colorcut = colors[0]*cut[0] + colors[1]*cut[1] > cut[2]
        i_clean = i_clean & i_makescut    
    return i_clean


def _pm_G_line(G):
    return 10**(0.4*(G-18.25))

def cut_pm_G(tab):
    i_makescut = tab['pm'] < _pm_G_line(tab['phot_g_mean_mag'])
    return i_makescut


# gets nearest neighbor first, then cuts by sep, so guaranteed to be 0 or 1 matches
def cross_match_nearest(ra1, dec1, ra2, dec2, separation):
    coords1 = SkyCoord(ra=ra1, dec=dec1, frame='icrs')    
    coords2 = SkyCoord(ra=ra2, dec=dec2, frame='icrs') 
    index_list_all, sep2d, _ = match_coordinates_sky(coords1, coords2, nthneighbor=1)
    idx_close = sep2d < separation
    # The indices that match_coordinates produces are into coord2; get only the ones with close match
    index_list_2in1 = index_list_all[idx_close]
    # index_list_all has shape coords1, so the locations of the close matches are where in coords1
    # the matches are
    index_list_1in2 = np.where(idx_close)[0]
    return index_list_1in2, index_list_2in1


# Cross match function using astropy
def cross_match(ra1, dec1, ra2, dec2, separation):
    coords1 = SkyCoord(ra=ra1, dec=dec1, frame='icrs')    
    coords2 = SkyCoord(ra=ra2, dec=dec2, frame='icrs') 
    cross = search_around_sky(coords1, coords2, separation) 
    index_list_1in2, index_list_2in1 = cross[0], cross[1] 
    return index_list_1in2, index_list_2in1


def add_spzs(tab_gaia, fn_spz='../data/redshifts_spz_kNN.fits'):
    tab_spz = Table.read(fn_spz, format='fits')
    assert np.allclose(tab_gaia['source_id'], tab_spz['source_id']), "Source IDs don't line up! They should by construction"
    tab_gaia.add_column(tab_spz['redshift_spz'], name='redshift_spz')
    tab_gaia.add_column(tab_spz['redshift_sdss'], name='redshift_sdss')
    

def redshift_cut_index(tab, z_min, redshift_key):
    #Include only SDSS quasars with z>z_min (this removes zeros and nans, and maybe a few others)
    idx_zgood = tab[redshift_key] > z_min
    return idx_zgood


def load_table(fn_fits, format='fits'):
    return Table.read(fn_fits, format=format)


def write_table(fn_table, data_cols, col_names, overwrite=False):
    tab = Table(data_cols, names=col_names)
    tab.write(fn_table, overwrite=overwrite)
    return tab


# copied from https://stackoverflow.com/questions/49372918/group-numpy-into-multiple-sub-arrays-using-an-array-of-values
def groupby(values, group_indices):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = group_indices.argsort(kind='mergesort')
    values_sorted = values[sidx]
    group_indices_sorted = group_indices[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,group_indices_sorted[1:] != group_indices_sorted[:-1],True])

    # Split input array with those start, stop ones
    values_grouped = [values_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return values_grouped, list(set(group_indices_sorted))


### Extinction

def fetch_dustmap(map_name='csfd', data_dir='../data/dustmaps'):
    # if already exists, don't need to fetch
    if os.path.exists(f'{data_dir}/{map_name}'):
        return
    import dustmaps
    import dustmaps.sfd, dustmaps.csfd
    map_dict = {'sfd': dustmaps.sfd, 
                'csfd': dustmaps.csfd}
    if map_name not in map_dict:
        raise ValueError(f"Map name {map_name} not recognized!")
    from dustmaps.config import config
    config['data_dir'] = data_dir

    map_dict[map_name].fetch()


def get_ebv(ra, dec, map_name='csfd'):
    assert map_name in ['sfd', 'csfd'], "Map name not recognized!"
    fetch_dustmap(map_name=map_name) #will only fetch if not already fetched
    if map_name=='sfd':
        sfd = SFDQuery()
    elif map_name=='csfd':
        sfd = CSFDQuery()
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs') 
    ebv_orig = sfd(coords)
    # rescaling correction described in https://arxiv.org/pdf/1009.4933.pdf
    # the rescaling factor is not included in CSFD either (see sec A1 of https://arxiv.org/pdf/2306.03926.pdf)
    ebv_rescaled = 0.86*ebv_orig
    return ebv_rescaled


def add_ebv(tab, map_name='csfd'):
    ebv = get_ebv(tab['ra'], tab['dec'], map_name=map_name)
    tab.add_column(ebv, name='ebv')


def get_extinction(ra, dec, R=3.1, map_name='csfd'):
    ebv = get_ebv(ra, dec, map_name=map_name)
    return R*ebv


def add_extinction(tab, R=3.1, map_name='csfd'):
    if 'ebv' in tab.columns:
        A_v = R*tab['ebv']
    else:
        A_v = get_extinction(tab['ra'], tab['dec'], R=R, map_name=map_name)
    tab.add_column(A_v, name='A_v')


### Coordinates

# following https://mathworld.wolfram.com/SphericalCoordinates.html
def cartesian_to_spherical(x, y, z):
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arctan2(y,x)
    phi     =  np.arccos(z/r)
    return np.array([r, theta, phi])


def spherical_to_cartesian(r, theta, phi):
    x       =  r*np.cos(theta)*np.sin(phi)
    y       =  r*np.sin(theta)*np.sin(phi)
    z       =  r*np.cos(phi)
    return np.array([x, y, z])

# changed this! fix surrounding code if this breaks
def spherical_to_radec(theta, phi):
    ra = theta * 180/np.pi #+ 180
    #dec = phi * 180/np.pi - 90  
    dec = 90 - phi * 180/np.pi
    return ra, dec 


# TODO: phi here isnt the inverse of above, check!
def radec_to_spherical(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi


def cartesian_to_radec(x, y, z):
    _, theta, phi = cartesian_to_spherical(x, y, z)
    return spherical_to_radec(theta, phi)


def radec_to_cartesian(r, ra, dec):    
    theta, phi = radec_to_spherical(ra, dec)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    return np.array([x, y, z])


def random_ra_dec_on_sphere(rng, N_sphere):
    us = rng.random(size=N_sphere)
    vs = rng.random(size=N_sphere)
    theta_sphere = 2 * np.pi * us
    phi_sphere = np.arccos(2*vs-1)
    
    ra_sphere, dec_sphere = spherical_to_radec(theta_sphere, phi_sphere)
    return ra_sphere*u.deg, dec_sphere*u.deg


### Units

def Mpc_to_Mpcperh(distances_Mpc, cosmo):
    return distances_Mpc * cosmo.h


def Mpcperh_to_Mpc(distances_Mpcperh, cosmo):
    return distances_Mpcperh / cosmo.h


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name=None):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    if name is not None:
        plt.register_cmap(cmap=newcmap)

    return newcmap


def split_train_val_test_idxs(random_ints, N_tot=None, frac_train=0.70, frac_val=0.15, frac_test=0.15):

    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    if N_tot is None:
        N_tot = len(random_ints)
    int_train = int(frac_train*N_tot)
    int_test = int((1-frac_test)*N_tot)

    idx_train = np.where(random_ints < int_train)[0]
    idx_test = np.where(random_ints >= int_test)[0]
    idx_val = np.where((random_ints >= int_train) & (random_ints < int_test))[0]

    return idx_train, idx_val, idx_test


def split_train_val_test(random_ints, N_tot=None, frac_train=None, frac_val=None, frac_test=None):

    if frac_test is None:
        frac_test = 1.0-frac_train-frac_val
    if frac_train is None:
        frac_train = 1.0-frac_val-frac_test
    if frac_val is None:
        frac_val = 1.0-frac_train-frac_test

    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    if N_tot is None:
        N_tot = len(random_ints)
    int_train = int(frac_train*N_tot)
    int_test = int((1-frac_test)*N_tot)

    i_train = random_ints < int_train
    i_val = (random_ints >= int_train) & (random_ints < int_test)
    i_test = random_ints >= int_test

    return i_train, i_val, i_test


def add_randints_column(tab):
    rng = np.random.default_rng(seed=42)
    tab['rand_ints'] = rng.choice(range(len(tab)), size=len(tab), replace=False)


def make_superset_cuts(tab):
    col_names_necessary = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                               'mag_w1_vg', 'mag_w2_vg', 'redshift_qsoc']
    tab = get_table_with_necessary(tab, col_names_necessary)
    G_cut = 20.6
    i_Gcut = tab['phot_g_mean_mag'] < G_cut
    tab = tab[i_Gcut]
    print(f"Removed {np.sum(~i_Gcut)} sources with G>={G_cut}")
    return tab


def get_table_with_necessary(tab, col_names_necessary):
        
    cols_necessary = []
    masks_necessary = []
    for col_name in col_names_necessary:
        cols_necessary.append(tab[col_name])
        masks_necessary.append(tab.mask[col_name])
    cols_necessary = np.array(cols_necessary).T
    masks_necessary = np.array(masks_necessary).T
    i_finite = np.all(np.isfinite(cols_necessary), axis=1)
    i_unmasked = np.all(~masks_necessary, axis=1)
    i_has_necessary = i_finite & i_unmasked
    print(f"N with necessary data: {np.sum(i_has_necessary)} ({np.sum(i_has_necessary)/len(i_has_necessary):.3f})")
    return tab[i_has_necessary]


### purity and completeness ###

def confusion_matrix(C_pred, C_true, labels, priors=None, class_fracs=None):

    C_pred = np.array(list(C_pred))
    C_true = np.array(list(C_true))

    N_classes = len(labels)
    conf_mat = np.empty((N_classes, N_classes))
    for i in range(N_classes):
        for j in range(N_classes):
            conf_mat[i, j] = np.sum((C_true==labels[i]) & (C_pred==labels[j]))
    
    if priors is not None and class_fracs is None:
        assert len(priors)==N_classes
        priors_norm = priors/np.sum(priors)
        for i in range(N_classes):
            conf_mat[i] *= priors_norm[i]
    
    if class_fracs is not None:
        assert len(class_fracs)==N_classes
        if priors is None:
            priors = np.ones(N_classes)
        priors_norm = priors/np.sum(priors)      
        class_fracs_norm = class_fracs/np.sum(class_fracs)
        
        denominator = np.sum(priors_norm/class_fracs_norm)
        lambdas = priors_norm/class_fracs_norm * 1/denominator
        for i in range(N_classes):
            conf_mat[i] *= lambdas[i]   
    
    return conf_mat

def N_TP(conf_mat, class_labels, label='q'):
    i = class_labels.index(label)
    TP = conf_mat[i,i]
    return TP

def N_FP(conf_mat, class_labels, label='q'):
    i = class_labels.index(label)
    FP = np.sum([conf_mat[j,i] for j in range(len(class_labels)) if j!=i])
    return FP

def N_FN(conf_mat, class_labels, label='q'):
    i = class_labels.index(label)
    FN = np.sum([conf_mat[i,j] for j in range(len(class_labels)) if j!=i])
    return FN

def purity(conf_mat, class_labels, label='q'):
    i = class_labels.index(label)
    TP = conf_mat[i,i]
    FP = np.sum([conf_mat[j,i] for j in range(len(class_labels)) if j!=i])
    return TP/(TP+FP)

def completeness(conf_mat, class_labels, label='q'):
    i = class_labels.index(label)
    TP = conf_mat[i,i]
    FN = np.sum([conf_mat[i,j] for j in range(len(class_labels)) if j!=i])
    return TP/(TP+FN)

# assumes all s_ids_pred_target are in target class (bc we're labeling all of our catalog 'quasars')
def get_classes(s_ids_pred_target, s_ids_valid, c_valid, target='q'):
    s_ids_pred_target = np.array(s_ids_pred_target)
    # Get which sources in validation set are in the predicted catalog
    i_in_pred = np.isin(s_ids_valid, s_ids_pred_target)
    print(np.sum(i_in_pred))
    # The predicted catalog is all targets; non-targets are other ('o')
    c_pred = np.full(len(c_valid), 'o')
    c_pred[i_in_pred] = target
    return c_pred


def compute_rmse(y_pred, y_true, fractional=False):
    diffs = y_pred - y_true
    if fractional:
        diffs /= y_true
    return np.sqrt(np.mean(diffs**2))


### table details

label2unit_dict = {'source_id': None,
        'unwise_objid': None,
        'redshift_quaia': None,
        'redshift_quaia_err': None,
        'ra': 'deg',
        'dec': 'deg', 
        'l': 'deg',
        'b': 'deg',
        'phot_g_mean_mag': 'mag',
        'phot_bp_mean_mag': 'mag',
        'phot_rp_mean_mag': 'mag',
        'mag_w1_vg': 'mag',
        'mag_w2_vg': 'mag',
        'pm': 'mas yr-1',
        'pmra': 'mas yr-1',
        'pmdec': 'mas yr-1',
        'pmra_error': 'mas yr-1',
        'pmdec_error': 'mas yr-1',
        }

unit2latex_dict = {'mas yr-1': 'mas yr$^{-1}$',
                   'None': ''}

label2description_dict = {'source_id': '\emph{Gaia} DR3 source identifier',
        'unwise_objid': 'unWISE DR1 source identifier',
        'redshift_quaia': 'spectrophotometric redshift estimate',
        'redshift_quaia_err': '$1\sigma$ uncertainty on spectrophotometric redshift estimate',
        'ra': 'right ascension',
        'dec': 'declination', 
        'l': 'galactic longitude',
        'b': 'galactic latitude',
        'phot_g_mean_mag': '\emph{Gaia} $G$-band mean magnitude',
        'phot_bp_mean_mag': '\emph{Gaia} integrated $BP$ mean magnitude',
        'phot_rp_mean_mag': '\emph{Gaia} integrated $RP$ mean magnitude',
        'mag_w1_vg': 'unWISE $W1$ magnitude',
        'mag_w2_vg': 'unWISE $W2$ magnitude',
        'pm': 'proper motion',
        'pmra': 'proper motion in right ascension direction',
        'pmdec': 'proper motion in declination direction',
        'pmra_error': 'standard error of proper motion in right ascension direction',
        'pmdec_error': 'standard error of proper motion in declination direction',
        }

label2format_dict = {'source_id': 'd',
            'unwise_objid': 's',
            'redshift_quaia': 'f',
            'redshift_quaia_err': 'f',
            'ra': 'f',
            'dec': 'f', 
            'l': 'f',
            'b': 'f',
            'phot_g_mean_mag': 'f',
            'phot_bp_mean_mag': 'f',
            'phot_rp_mean_mag': 'f',
            'mag_w1_vg': 'f',
            'mag_w2_vg': 'f',
            'pm': 'f',
            'pmra': 'f',
            'pmdec': 'f',
            'pmra_error': 'f',
            'pmdec_error': 'f',
            }

label2symbol_dict = {'source_id': '',
            'unwise_objid': '',
            'redshift_quaia': '$z_\mathrm{Quaia}$',
            'redshift_quaia_err': '',
            'ra': '',
            'dec': '', 
            'l': '',
            'b': '',
            'phot_g_mean_mag': '$G$',
            'phot_bp_mean_mag': '$BP$',
            'phot_rp_mean_mag': '$RP$',
            'mag_w1_vg': '$W1$',
            'mag_w2_vg': '$W2$',
            'pm': '$\mu$',
            'pmra': '$\mu_{\\alpha*}$',
            'pmdec': '$\mu_{\delta}$',
            'pmra_error': '$\sigma_{\mu\\alpha*}$',
            'pmdec_error': '$\sigma_{\mu\delta}$',
            }

