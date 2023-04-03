import numpy as np
import os
from numpy.random import default_rng

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join

import sklearn
from dustmaps.sfd import SFDQuery
from sklearn.neighbors import KDTree

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
#import xgboost as xgb
#from xgboost import XGBRegressor

import utils


def main():

    #Ks = np.arange(3, 34, 2)
    #Ks = np.arange(35, 50, 2)
    # Ks = [27]
    # for K in Ks:
    #     run(K=K)
    #     fn_spz_labeled = f'../data/redshift_estimates/redshifts_spz_labeled_kNN_K{K}.fits'
    #     combine_with_gaia_redshifts(fn_spz_labeled)
    #     fn_spz = f'../data/redshift_estimates/redshifts_spz_kNN_K{K}.fits'
    #     combine_with_gaia_redshifts(fn_spz)

    #K = 11
    K = 27
    run(K=K)
    fn_spz_labeled = f'../data/redshift_estimates/redshifts_spz_labeled_kNN_K{K}_std.fits'
    combine_with_gaia_redshifts(fn_spz_labeled)
    fn_spz = f'../data/redshift_estimates/redshifts_spz_kNN_K{K}_std.fits'
    combine_with_gaia_redshifts(fn_spz)


def run(K=27):

    rng = default_rng()

    mode = 'regression'
    redshift_estimator_name = 'kNN'
    #redshift_estimator_name = 'hgboost'
    #redshift_estimator_name = 'xgboost'
    #K = 31
    learning_rate = 0.005
    apply_to_all = True
    overwrite_model = True
    overwrite_table = True
    save_tables = True
    predict_residual = False

    # include prev?
    #fn_prev_estimate = f'../data/redshift_estimates/redshifts_spz_labeled_hgboost_scale_wphot.fits'
    prev_tag = None
    #prev_tag = '_hgboost_scale_wphot'
    #prev_tag = '_kNN_K27'
    #fn_prev_estimate = None
    prev_mode = 'add'
    spz_prev_name = 'redshift_spz_raw'

    save_tag_model = f'_K{K}_std'
    #save_tag_model = f'_K{K}_prev_hgboost'
    #save_tag_model = f'_K{K}_resid'
    #save_tag_model = f'_scale_wphot'
    #save_tag_model = f'_scale_wphot_prev_kNN_K27'


    # Save file names
    fn_model = f'../data/redshift_models/model_spz_{redshift_estimator_name}{save_tag_model}.fits'
    fn_spz_labeled = f'../data/redshift_estimates/redshifts_spz_labeled_{redshift_estimator_name}{save_tag_model}.fits'
    fn_spz = f'../data/redshift_estimates/redshifts_spz_{redshift_estimator_name}{save_tag_model}.fits'

    # Data file names
    fn_labeled = '../data/quasars_sdss_clean.fits'
    fn_gaia = '../data/gaia_candidates_clean.fits'

    redshift_estimator_dict = {'kNN': RedshiftEstimatorkNN,
                               'hgboost': RedshiftEstimatorHGBoost,
                               'gboost': RedshiftEstimatorGBoost,
                               'xgboost': RedshiftEstimatorXGBoost,
                               #'ANN': nnspz.RedshiftEstimatorANN, 
                            #    'ANN2class': RedshiftEstimatorANN2class,
                            #    'ANNmulticlass': RedshiftEstimatorANNmulticlass,
                            #    'SVM': RedshiftEstimatorSVM
                               }
    redshift_estimator_kwargs_dict = {'kNN': {'K': K},
                                      'hgboost': {},
                                      'gboost': {},
                                      'xgboost': {},
                                    #  'ANN': {'rng': rng, 'learning_rate': learning_rate},
                                    #   'ANN2class': {'rng': rng, 'learning_rate': learning_rate_classifier},
                                    #   'ANNmulticlass': {'rng': rng, 'learning_rate': learning_rate_classifier, 
                                    #                     'N_classes': N_classes},
                                    #   'SVM': {'C':1e4, 'gamma': 0.1}
                                      }
    redshift_estimator_class = redshift_estimator_dict[redshift_estimator_name]                        
    redshift_estimator_kwargs = redshift_estimator_kwargs_dict[redshift_estimator_name]

    tab_labeled = utils.load_table(fn_labeled)
    print(tab_labeled.columns)

    # Construct full feature matrix
    print("Constructing feature matrix")
    #feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2', 'phot_g_mean_mag']
    feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2', 'phot_g_mean_mag',
                    'phot_bp_mean_mag', 'phot_rp_mean_mag', 'mag_w1_vg', 'mag_w2_vg'
                    ]
    if prev_tag is not None:
        fn_prev_estimate_labeled = f'../data/redshift_estimates/redshifts_spz_labeled{prev_tag}.fits'
        tab_spz_prev_labeled = Table.read(fn_prev_estimate_labeled, format='fits')
        tab_spz_prev_labeled.keep_columns(['source_id', spz_prev_name])
        tab_labeled = astropy.table.join(tab_labeled, tab_spz_prev_labeled, keys='source_id', join_type='left')
        if apply_to_all:
            fn_prev_estimate = f'../data/redshift_estimates/redshifts_spz{prev_tag}.fits'
            tab_spz_prev = Table.read(fn_prev_estimate, format='fits')
            tab_spz_prev.keep_columns(['source_id', spz_prev_name])
            tab_gaia = astropy.table.join(tab_gaia, tab_spz_prev, keys='source_id', join_type='left')
        if prev_mode=='replace_qsoc':
            print(f"Replacing QSOC with SPZ from prev estimate, {prev_tag}")
            i_qsoc = feature_keys.index('redshift_qsoc')
            feature_keys[i_qsoc] = spz_prev_name    
        elif prev_mode=='add':
            print(f"Adding SPZ from prev estimate to features {prev_tag}")
            feature_keys.append(spz_prev_name)
    print('feature keys:', feature_keys)

    X_labeled = construct_X(tab_labeled, feature_keys)
    Y_labeled = tab_labeled['z_sdss']
    Y_qsoc_labeled = tab_labeled['redshift_qsoc']
    N_labeled = X_labeled.shape[0]

    # this is essentially shuffling an array of 1-N
    #random_ints = rng.choice(range(N_labeled), size=N_labeled, replace=False)
    rand_ints_labeled = tab_labeled['rand_ints']
    # N_tot=N_gaia because the numbers go up to all the ones in the clean catalog
    i_train, i_valid, i_test = utils.split_train_val_test(rand_ints_labeled,
                                     frac_train=0.7, frac_val=0.15, frac_test=0.15)

    print(f"N_train: {np.sum(i_train)} ({np.sum(i_train)/N_labeled:.3f})") 
    print(f"N_valid: {np.sum(i_valid)} ({np.sum(i_valid)/N_labeled:.3f})") 
    print(f"N_test: {np.sum(i_test)} ({np.sum(i_test)/N_labeled:.3f})") 

    # when should be using validation vs test?? 
    # think test should only be for final plots and numbers,
    # which will be in a notebook, not here
    X_train = X_labeled[i_train]
    X_valid = X_labeled[i_valid]
    #X_test = X_labeled[i_test]

    Y_train = Y_labeled[i_train]
    Y_valid = Y_labeled[i_valid]
    #Y_test = Y_labeled[i_test]

    # only need QSOC for valid to compare our results
    Y_qsoc_valid = Y_qsoc_labeled[i_valid]
    Y_current_train = None
    if predict_residual:
        Y_current_train = Y_qsoc_labeled[i_train]

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    assert X_train.shape[0]==Y_train.shape[0], "X and Y must have same length!"

    if apply_to_all:
        # Load data
        print("Loading data")
        tab_gaia = utils.load_table(fn_gaia)
        N_gaia = len(tab_gaia)
        print(f"N of clean gaia catalog: {N_gaia}")
        X_gaia = construct_X(tab_gaia, feature_keys)
        Y_qsoc_gaia = tab_gaia['redshift_qsoc']

    # Run redshift estimation
    print("Running redshift estimation")
    # cross_validate(redshift_estimator_class, 
    #                X_train, Y_train,
    #                redshift_estimator_kwargs, rng)

    

    if mode=='regression':
        if os.path.exists(fn_model) and not overwrite_model:
            print(f"Model {fn_model} already exists and overwrite_model={overwrite_model}")
            print("So load it in!")
            redshift_estimator = redshift_estimator_class(train_mode=False, test_mode=True)
            redshift_estimator.load_model(fn_model)
        else:
            print(f"Training model {fn_model} (overwrite_model={overwrite_model})")                                                             
            redshift_estimator = redshift_estimator_class(X_train=X_train,          
                                                    Y_train=Y_train, 
                                                    X_valid=X_valid, 
                                                    Y_valid=Y_valid,
                                                    Y_current_train=Y_current_train,
                                                    **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.save_model(fn_model)

        # Apply
        if predict_residual:
            Y_hat_labeled, sigma_z_labeled = redshift_estimator.predict_residual(X_labeled, Y_qsoc_labeled)
            if apply_to_all:
                Y_hat_gaia, sigma_z_gaia = redshift_estimator.predict_residual(X_gaia, Y_qsoc_gaia)

        else:
            Y_hat_labeled, sigma_z_labeled = redshift_estimator.predict(X_labeled)
            if apply_to_all:
                Y_hat_gaia, sigma_z_gaia = redshift_estimator.predict(X_gaia)
       
        print(sigma_z_gaia)

    else:
        raise ValueError("MODE NOT RECOGNIZED")

    Y_hat_valid = Y_hat_labeled[i_valid]
    # Print results
    dzs = [0.01, 0.1, 0.2, 1.0]
    for dz in dzs:
        print(f"Fraction recovered with Dz/(1+z)<{dz}:")
        frac_recovered = utils.get_fraction_recovered(Y_valid, Y_hat_valid, dz)
        print(f"SPZ: {frac_recovered:.3f}")
        frac_recovered_qsoc = utils.get_fraction_recovered(Y_valid, Y_qsoc_valid, dz)
        print(f"QSOC: {frac_recovered_qsoc:.3f}")

    # Save results
    if save_tables:
        print("Saving results on labeled data")
        columns_to_keep = ['source_id', 'redshift_qsoc','z_sdss',
                           'phot_g_mean_mag', 'rand_ints']
        tab_labeled.keep_columns(columns_to_keep)

        tab_labeled['redshift_spz_raw'] = Y_hat_labeled
        tab_labeled['redshift_spz_err'] = sigma_z_labeled
        tab_labeled.write(fn_spz_labeled, overwrite=overwrite_table)
        print(f"Wrote specphotozs to {fn_spz_labeled}!")

    if save_tables and apply_to_all:
        print("Saving results on Gaia catalog")
        columns_to_keep = ['source_id', 'redshift_qsoc',
                           'phot_g_mean_mag', 'rand_ints']        
        tab_gaia.keep_columns(columns_to_keep)

        tab_gaia['redshift_spz_raw'] = Y_hat_gaia
        tab_gaia['redshift_spz_err'] = sigma_z_gaia
        print(tab_gaia['redshift_spz_err'])
        tab_gaia.write(fn_spz, overwrite=overwrite_table)
        print(f"Wrote specphotozs to {fn_spz}!")


def construct_X(tab, feature_keys):

    X = []
    for feature_key in feature_keys:
        X.append(tab[feature_key])
    X = np.array(X).T
    i_badfeat = np.any(~np.isfinite(X), axis=1)
    # shouldn't be any bad features because we cleaned up first in make_data_tables.gaia_clean()
    assert np.sum(i_badfeat)==0, "Some bad feature data in clean catalog!"
    # TODO: put back this assert when doing real apply; removing for now bc only test
    return X


def combine_with_gaia_redshifts(fn_spz):

    tab_spz = utils.load_table(fn_spz)

    # rand_ints_labeled = tab_spz['rand_ints']
    # # N_tot=N_gaia because the numbers go up to all the ones in the clean catalog
    # i_train, i_valid, i_test = utils.split_train_val_test(rand_ints_labeled,
    #                                 frac_train=0.7, frac_val=0.15, frac_test=0.15)

    z_spzraw = tab_spz['redshift_spz_raw']#[i_valid]
    z_gaia = tab_spz['redshift_qsoc']#[i_valid]

    dz_min = 0.05
    dz_max = 0.1

    dz_spzraw_gaia = (z_spzraw - z_gaia)/(1 + z_gaia)
    frac_shift = 1.0-(np.abs(dz_spzraw_gaia)-dz_min)/(dz_max-dz_min)
    frac_shift[frac_shift<0.0] = 0.0
    frac_shift[frac_shift>1.0] = 1.0

    z_spz = z_spzraw - frac_shift*dz_spzraw_gaia*(1+z_gaia)
    # im = np.where((np.abs(dz_spzraw_gaia) > 0.15) & (np.abs(dz_spzraw_gaia) < 0.1))[0][0]
    # print(z_spzraw[im], z_gaia[im], tab_spz['z_sdss'][im])
    # print(frac_shift[im], dz_spzraw_gaia[im], (frac_shift*dz_spzraw_gaia)[0])
    # print(z_spz[im])

    # Print results
    if False:
        z_sdss = tab_spz['z_sdss'][i_valid]
        dzs = [0.01, 0.1, 0.2, 1.0]
        for dz in dzs:
            print(f"Fraction recovered with Dz/(1+z)<{dz}:")
            frac_recovered = utils.get_fraction_recovered(z_sdss, z_spz, dz)
            print(f"SPZ: {frac_recovered:.3f}")
            frac_recovered = utils.get_fraction_recovered(z_sdss, z_spzraw, dz)
            print(f"SPZraw: {frac_recovered:.3f}")
            frac_recovered_qsoc = utils.get_fraction_recovered(z_sdss, z_gaia, dz)
            print(f"QSOC: {frac_recovered_qsoc:.3f}")

    tab_spz['redshift_spz'] = z_spz
    tab_spz.write(fn_spz, overwrite=True)
    print(f"Added SPZ/Gaia smoothed redshifts to {fn_spz}")



def cross_validate(redshift_estimator_class,
                   X_train, Y_train,
                   redshift_estimator_kwargs, rng, n_samples=8):

    z_errs_close = [0.1, 0.2]

    print("Cross validating")
    i_sample_vals = np.arange(n_samples)
    # high is exclusive
    i_samples_loo = rng.integers(low=0, high=n_samples, size=X_train.shape[0])

    Y_hat = np.empty(X_train.shape[0])
    sigma_z = np.empty(X_train.shape[0])

    print("ONE CROSS VAL FOR NOW")
    #for i_sample_val in i_sample_vals:
    for i_sample_val in [i_sample_vals[0]]:
        print(f"Leave-one-out sample {i_sample_val}")
        idx_train = i_samples_loo != i_sample_val
        idx_test = i_samples_loo == i_sample_val
        X_train_loo, Y_train_loo = X_train[idx_train], Y_train[idx_train]
        X_test_loo, Y_test_loo = X_train[idx_test], Y_train[idx_test]

        redshift_estimator = redshift_estimator_class(X_train_loo, Y_train_loo, X_test_loo,
                                                      **redshift_estimator_kwargs)

        redshift_estimator.train()
        Y_hat_test_loo, sigma_z_test_loo = redshift_estimator.apply()

        #tree_loo = self.build_tree(X_train_loo)
        # TODO: only passing X train to check things, can delete that kwarg
        #Y_hat_valid_loo, sigma_z_valid_loo = self.get_median_kNNs(X_valid_loo, Y_train=Y_train_loo)
        
        Y_hat[idx_test] = Y_hat_test_loo
        sigma_z[idx_test] = sigma_z_test_loo

        for z_err_close in z_errs_close:
            frac_recovered = get_fraction_recovered(Y_test_loo, Y_hat_test_loo, z_err_close)
            print(rf"Fraction recovered with $\delta z$<{z_err_close}: {frac_recovered:.3f}")

    for z_err_close in z_errs_close:
        frac_recovered = get_fraction_recovered(Y_train, Y_hat, z_err_close)
        print(rf"Overall fraction recovered with $\delta z$<{z_err_close}: {frac_recovered:.3f}")


def get_fraction_recovered(Y_true, Y_hat, z_err_close):
    return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)



class RedshiftEstimator():

    def __init__(self, X_train=None, Y_train=None, 
                       X_valid=None, Y_valid=None, 
                       X_apply=None, 
                       train_mode=True, test_mode=False,
                       Y_current_train=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_apply = X_apply
        self.Y_current_train = Y_current_train
        self.train_mode = train_mode
        self.test_mode = test_mode
        if Y_current_train is not None:
            self.Y_train = Y_train - Y_current_train


    def train(self):
        pass 


    def predict(self, X_input):
        pass

    def predict_residual(self, X_input, Y_current_input):
        Y_resid_pred, sigma_z = self.predict(X_input)
        Y_hat = Y_current_input + Y_resid_pred
        return Y_hat, sigma_z


    def apply(self):
        pass


    def save_model(self, fn_model):
        pass


    def load_model(self, fn_model):
        pass



class RedshiftEstimatorkNN(RedshiftEstimator):

    
    def __init__(self, *args, K=11, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.scale_x()


    def scale_x(self):
        # mean and stdev
        #self.scaler_x = StandardScaler() # TODO revisit !! 
        #does nothing while keeping notation consistent
        self.scaler_x = StandardScaler(with_mean=False, with_std=False) 
        # scales all besides 1st column, QSOC
        #N_feat = self.X_train.shape[1]
        #self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        self.scaler_x.fit(self.X_train)
        self.X_train_scaled = self.scaler_x.transform(self.X_train)


    def train(self):
        print("Training")
        self.tree_train = self.build_tree(self.X_train_scaled)


    def apply(self):
        print("Applying")
        return self.predict(self.X_apply)


    def build_tree(self, X):
        print("Building kdTree")
        return KDTree(X)


    def predict(self, X_input):
        X_input_scaled = self.scaler_x.transform(X_input)

        print("Getting median Z of nearest neighbors")
        print("K =", self.K)
        dists, inds = self.tree_train.query(X_input_scaled, k=self.K+1)
        # if nearest neighbor is itself (dist~0), exclude that one;
        # to do this, need to get more neighbors than maybe necessary
        # to keep it at K overall
        dist_min = 1e-8 #hack
        idx_nearest_dist0 = dists[:,0] < dist_min
        print(f"A fraction {np.sum(idx_nearest_dist0)/len(idx_nearest_dist0):.3f} of objects have nearest neighbor w dist zero; cutting these from median")
        inds_nodist0 = np.empty((inds.shape[0], self.K), dtype=int)
        inds_nodist0[idx_nearest_dist0] = inds[idx_nearest_dist0,1:]
        inds_nodist0[~idx_nearest_dist0] = inds[~idx_nearest_dist0,:-1]
        #low_z, Y_hat, up_z = np.percentile(self.Y_train[inds_nodist0], (2.5, 50, 97.5), axis=1)
        #sigma_z = (up_z - low_z)/4
        low_z, Y_hat, up_z = np.percentile(self.Y_train[inds_nodist0], (16, 50, 84), axis=1)
        sigma_z = 0.5*(up_z - low_z)
        return Y_hat, sigma_z



class RedshiftEstimatorHGBoost(RedshiftEstimator):

    def __init__(self, *args, max_iter=1000, learning_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.scale_x()


    def scale_x(self):
        # TODO revisit !! 
        # mean and stdev
        self.scaler_x = StandardScaler() 
        #does nothing while keeping notation consistent
        #self.scaler_x = StandardScaler(with_mean=False, with_std=False) 
        # scales all besides 1st column, QSOC
        #N_feat = self.X_train.shape[1]
        #self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        self.scaler_x.fit(self.X_train)
        self.X_train_scaled = self.scaler_x.transform(self.X_train)


    def train(self):
        print("Training")
        self.model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate)
        self.model.fit(self.X_train_scaled, self.Y_train)


    def apply(self):
        print("Applying")
        return self.predict(self.X_apply)


    def predict(self, X_input):
        X_input_scaled = self.scaler_x.transform(X_input)

        Y_hat = self.model.predict(X_input_scaled)
        sigma_z = np.full(len(Y_hat), np.nan)

        return Y_hat, sigma_z


class RedshiftEstimatorGBoost(RedshiftEstimator):

    def __init__(self, *args, n_estimators=1000, learning_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.scale_x()


    def scale_x(self):
        # mean and stdev
        #self.scaler_x = StandardScaler() # TODO revisit !! 
        #does nothing while keeping notation consistent
        self.scaler_x = StandardScaler(with_mean=False, with_std=False) 
        # scales all besides 1st column, QSOC
        #N_feat = self.X_train.shape[1]
        #self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        self.scaler_x.fit(self.X_train)
        self.X_train_scaled = self.scaler_x.transform(self.X_train)


    def train(self):
        print("Training")
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        self.model.fit(self.X_train_scaled, self.Y_train)


    def apply(self):
        print("Applying")
        return self.predict(self.X_apply)


    def predict(self, X_input):
        X_input_scaled = self.scaler_x.transform(X_input)

        Y_hat = self.model.predict(X_input_scaled)
        sigma_z = np.full(len(Y_hat), np.nan)

        return Y_hat, sigma_z



class RedshiftEstimatorXGBoost(RedshiftEstimator):

    def __init__(self, *args, n_estimators=500, learning_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.scale_x()


    def scale_x(self):
        # mean and stdev
        self.scaler_x = StandardScaler() # TODO revisit !! 
        #does nothing while keeping notation consistent
        #self.scaler_x = StandardScaler(with_mean=False, with_std=False) 
        # scales all besides 1st column, QSOC
        #N_feat = self.X_train.shape[1]
        #self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        self.scaler_x.fit(self.X_train)
        self.X_train_scaled = self.scaler_x.transform(self.X_train)


    def train(self):
        print("Training")

        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 3}  # the number of classes that exist in this datset
        num_round = 20  # the number of training iterations

        #dtrain = xgb.DMatrix(self.X_train_scaled, label=Y_train)
        #dtrain = xgb.DMatrix(self.X_train_scaled)
        #bst = xgb.train(param, dtrain, num_round)

        Y_train = np.array(list(self.Y_train))
        self.model = XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=10)
        self.model.fit(np.array(self.X_train_scaled), Y_train)


    def apply(self):
        print("Applying")
        return self.predict(self.X_apply)


    def predict(self, X_input):
        X_input_scaled = self.scaler_x.transform(X_input)

        Y_hat = self.model.predict(X_input_scaled)
        sigma_z = np.full(len(Y_hat), np.nan)

        return Y_hat, sigma_z



if __name__=='__main__':
    main()
