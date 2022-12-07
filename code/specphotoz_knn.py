import numpy as np
from numpy.random import default_rng

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join

from dustmaps.sfd import SFDQuery
from sklearn.neighbors import KDTree

import utils


def main():

    G_max = 20.5

    # save name
    fn_spz = f'../data/redshifts_spz_kNN_G{G_max}_noself.fits'
    overwrite = True

    # Load data
    print("Loading data")
    fn_gaia = '../data/gaia_slim.fits'
    tab_gaia = utils.load_table(fn_gaia)
    # TEST ONLY W SMALL AMOUNT
    #tab_gaia = tab_gaia[np.random.randint(0, len(tab_gaia), size=10000)]
    N_gaia = len(tab_gaia)
    print(f"Number of Gaia QSO candidates: {N_gaia}")

    fn_sdss = '../data/sdss_slim.fits'
    tab_sdss = utils.load_table(fn_sdss)

    z_min = 0.01 #magic #hyperparameter
    redshift_key = 'Z'
    idx_zgood = redshift_cut_index(tab_sdss, z_min, redshift_key)
    tab_sdss = tab_sdss[idx_zgood]
    print(f"Number of SDSS QSOs:, {len(tab_sdss)}")

    print("Constructing feature matrix")
    # Get reddening
    if 'ebv' not in tab_gaia.columns:
        utils.add_ebv(tab_gaia)

    # color cuts
    gaia_wise_colors = ['g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2']
    if np.any(np.in1d(gaia_wise_colors, tab_gaia.columns)):
        utils.add_gaia_wise_colors(tab_gaia)
    color_cuts = [[0., 1., 0.2], [1., 1., 2.9]]
    idx_clean_gaia = cuts_index(tab_gaia, G_max, color_cuts) 
    print("N_clean:", np.sum(idx_clean_gaia))

    # Construct full feature matrix
    feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2', 'phot_g_mean_mag']
    X_gaia, idx_goodfeat = construct_X(tab_gaia, feature_keys)
    # these indexes are the ones we will estimate SPZs for
    idx_withspzs = idx_clean_gaia & idx_goodfeat
    print(len(tab_gaia), len(idx_clean_gaia), len(idx_goodfeat), len(idx_withspzs))

    X_gaia_withspzs = X_gaia[idx_withspzs]

    # make table with just the ones we will estimate SPZs for
    tab_gaia_withspzs = tab_gaia[idx_withspzs]

    # Cross-match
    print("Performing cross-match")
    separation = 1*u.arcsec
    index_list_gaiaINsdss, index_list_sdssINgaia = cross_match(tab_gaia_withspzs, 
                                           tab_gaia_withspzs['ra'], tab_gaia_withspzs['dec'],
                                           tab_sdss, tab_sdss['RA']*u.degree, tab_sdss['DEC']*u.degree,
                                           separation=separation)

    # Split training (where have redshifts) and not
    X_train = X_gaia_withspzs[index_list_gaiaINsdss]
    # Apply to all, including those with SDSS redshifts
    # (TODO: is this correct??)
    X_apply = X_gaia_withspzs

    Y_train = tab_sdss[index_list_sdssINgaia]['Z']
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_apply: {X_apply.shape}")
    assert X_train.shape[0]==Y_train.shape[0], "X and Y must have same length!"

    # Run kNN
    print("Running kNN")
    K = 11 #hyperparameter
    rng = default_rng()
    knn = spz_kNN(X_train, Y_train, X_apply, K=K)
    knn.cross_validate(rng)
    knn.train()
    knn.apply()

    # create new table and save
    print("Creating results table")
    redshift_spz = np.full(N_gaia, np.nan)
    redshift_spz[idx_withspzs] = knn.Y_hat_apply

    redshift_spz_err = np.full(N_gaia, np.nan)
    redshift_spz_err[idx_withspzs] = knn.sigma_z

    redshift_sdss = np.full(N_gaia, np.nan)
    # not sure why this requires multiple steps but it does!!
    redshift_sdss_withspzs = redshift_sdss[idx_withspzs]
    redshift_sdss_withspzs[index_list_gaiaINsdss] = Y_train
    redshift_sdss[idx_withspzs] = redshift_sdss_withspzs

    data_cols = [tab_gaia['source_id'], redshift_spz, redshift_spz_err, redshift_sdss]
    col_names = ('source_id', 'redshift_spz', 'redshift_spz_err', 'redshift_sdss')
    utils.write_table(fn_spz, data_cols, col_names, overwrite=overwrite)
    print(f"Wrote specphotozs to {fn_spz}!")


def cuts_index(tab, G_max, color_cuts):

    # start with all
    idx_clean = np.full(len(tab), True)

    # g magnitude cut
    idx_gmagcut = (tab['phot_g_mean_mag'] < G_max)
    idx_clean = idx_clean & idx_gmagcut

    # color cuts
    for cut in color_cuts:
        idx_colorcut = gw1_w1w2_cut_index(tab['g_w1'], tab['w1_w2'], cut)
        idx_clean = idx_clean & idx_colorcut
    
    print(f'Fraction of Gaia-SDSS cross-matched quasars that make cuts: {np.sum(idx_clean)/len(idx_clean):.3f}')
    return idx_clean




def redshift_cut_index(tab, z_min, redshift_key):
    #Include only SDSS quasars with z>z_min (this removes zeros and nans, and maybe a few others)
    idx_zgood = tab[redshift_key] > z_min
    return idx_zgood


# gets nearest neighbor first, then cuts by sep, so guaranteed to be 0 or 1 matches
def cross_match(tab1, ra1, dec1,
                tab2, ra2, dec2, separation):
    coords1 = SkyCoord(ra=ra1, dec=dec1, frame='icrs')    
    coords2 = SkyCoord(ra=ra2, dec=dec2, frame='icrs') 
    index_list_all, sep2d, _ = astropy.coordinates.match_coordinates_sky(coords1, coords2, nthneighbor=1)
    idx_close = sep2d < separation
    # The indices that match_coordinates produces are into coord2; get only the ones with close match
    index_list_2in1 = index_list_all[idx_close]
    # index_list_all has shape coords1, so the locations of the close matches are where in coords1
    # the matches are
    index_list_1in2 = np.where(idx_close)[0]
    return index_list_1in2, index_list_2in1


def gw1_w1w2_cut_index(g_w1, w1_w2, cut):
    return cut[0] * g_w1 + cut[1] * w1_w2 > cut[2]


def construct_X(tab, feature_keys):

    X = []
    for feature_key in feature_keys:
        X.append(tab[feature_key])
    X = np.array(X).T
    idx_goodfeat = np.all(np.isfinite(X), axis=1)
    
    return X, idx_goodfeat


class spz_kNN():

    def __init__(self, X_train, Y_train,
                       X_apply, K=11):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_apply = X_apply
        self.K = K


    def cross_validate(self, rng, n_samples=8):
        print("Cross validating")
        i_sample_vals = np.arange(n_samples)
        # high is exclusive
        i_samples_loo = rng.integers(low=0, high=n_samples, size=self.X_train.shape[0])

        self.Y_hat_valid = np.empty(self.X_train.shape[0])
        self.sigma_z_valid = np.empty(self.X_train.shape[0])

        for i_sample_val in i_sample_vals:
            print(f"Leave-one-out sample {i_sample_val}")
            idx_train = i_samples_loo != i_sample_val
            idx_valid = i_samples_loo == i_sample_val
            X_train_loo, Y_train_loo = self.X_train[idx_train], self.Y_train[idx_train]
            X_valid_loo = self.X_train[idx_valid]

            # construct tree
            print(X_train_loo.shape)
            tree_loo = self.build_tree(X_train_loo)
            # TODO: only passing X train to check things, can delete that kwarg
            Y_hat_valid_loo, sigma_z_valid_loo = self.get_median_kNNs(X_valid_loo, Y_train_loo, tree_loo, X_train=X_train_loo)
            
            self.Y_hat_valid[idx_valid] = Y_hat_valid_loo
            self.sigma_z_valid[idx_valid] = sigma_z_valid_loo

        z_errs_close = [0.1, 0.2]
        for z_err_close in z_errs_close:
            frac_recovered = self.get_fraction_recovered(self.Y_train, self.Y_hat_valid, z_err_close)
            print(rf"Fraction recovered with $\delta z$<{z_err_close}: {frac_recovered:.3f}")


    def get_fraction_recovered(self, Y_true, Y_hat, z_err_close):
        return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)


    def train(self):
        print("Training")
        self.tree_train = self.build_tree(self.X_train)


    def apply(self):
        print("Applying")
        self.Y_hat_apply, self.sigma_z = self.get_median_kNNs(self.X_apply, self.Y_train, self.tree_train)


    def build_tree(self, X_train):
        print("Building kdTree")
        return KDTree(X_train)


    def get_median_kNNs(self, X_apply, Y_train, tree, X_train=None):
        print("Getting median Z of nearest neighbors")
        dists, inds = tree.query(X_apply, k=self.K+1)
        # if nearest neighbor is itself (dist~0), exclude that one;
        # to do this, need to get more neighbors than maybe necessary
        # to keep it at K overall
        dist_min = 1e-8 #hack
        idx_nearest_dist0 = dists[:,0] < dist_min
        print(f"A fraction {np.sum(idx_nearest_dist0)/len(idx_nearest_dist0):.3f} of objects have nearest neighbor w dist zero; cutting these from median")
        inds_nodist0 = np.empty((inds.shape[0], self.K), dtype=int)
        inds_nodist0[idx_nearest_dist0] = inds[idx_nearest_dist0,1:]
        inds_nodist0[~idx_nearest_dist0] = inds[~idx_nearest_dist0,:-1]
        low_z, Y_hat, up_z = np.percentile(Y_train[inds_nodist0], (2.5, 50, 97.5), axis=1)
        sigma_z = (up_z - low_z)/4
        return Y_hat, sigma_z



if __name__=='__main__':
    main()