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

import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer

import utils


def main():

    # How many GPUs are there?
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    G_max = 20.5
    rng = default_rng()
    #redshift_estimator_name = 'kNN'
    #mode = '2step'
    #mode = 'regression'
    #mode = 'zbins'
    mode = 'outliers'
    redshift_estimator_name_classifier = 'ANN2class'
    #redshift_estimator_name_classifier = 'SVM'
    #redshift_estimator_name_classifier = 'ANNmulticlass'
    #redshift_estimator_name = 'ANN'
    redshift_estimator_name = 'kNN'
    learning_rate_classifier = 0.005
    learning_rate = 0.005
    overwrite_classifier = True
    overwrite_model = True
    overwrite_table = True
    #fn_prev_estimate = f'../data/redshift_estimates/redshifts_spz_kNN_G20.5_valid.fits'
    #fn_prev_estimate = f'../data/redshift_estimates/redshifts_spz_kNN_G20.5_wspzqsoc.fits'
    fn_prev_estimate = None
    N_classes = 3

    # save names
    save_tag_model = f'_{mode}_wmags'
    #save_tag_model = '_zbins'
    #save_tag_model = '_withkNNspz'
    #save_tag_model = '_2step_trainonbad'
    save_tag_classifier = f'_{mode}_lr{learning_rate_classifier}_wmags'#_dz0.2line'
    #save_tag_classifier = f'_zbins{N_classes}nonuniform_lr{learning_rate_classifier}'
    #save_tag_classifier = f'_lr{learning_rate_classifier}'
    #save_tag = f'_lr{learning_rate}_valid'
    #save_tag = '_scaledNOqsoc'

    redshift_estimator_dict = {'kNN': RedshiftEstimatorkNN,
                               'ANN': RedshiftEstimatorANN, 
                               'ANN2class': RedshiftEstimatorANN2class,
                               'ANNmulticlass': RedshiftEstimatorANNmulticlass,
                               'SVM': RedshiftEstimatorSVM
                               }
    redshift_estimator_kwargs_dict = {'kNN': {'K': 11},
                                      'ANN': {'rng': rng, 'learning_rate': learning_rate},
                                      'ANN2class': {'rng': rng, 'learning_rate': learning_rate_classifier},
                                      'ANNmulticlass': {'rng': rng, 'learning_rate': learning_rate_classifier, 
                                                        'N_classes': N_classes},
                                      'SVM': {'C':1e4, 'gamma': 0.1}
                                      }
    redshift_estimator_class = redshift_estimator_dict[redshift_estimator_name]                        
    redshift_estimator_kwargs = redshift_estimator_kwargs_dict[redshift_estimator_name]

    fn_model = f'../data/redshift_models/model_spz_{redshift_estimator_name}_G{G_max}{save_tag_model}.fits'
    fn_model_classifier = f'../data/redshift_models/model_classifier_spz_{redshift_estimator_name_classifier}_G{G_max}{save_tag_classifier}.fits'
    fn_spz = f'../data/redshift_estimates/redshifts_spz_{redshift_estimator_name}_G{G_max}{save_tag_model}{save_tag_classifier}.fits'

    # Load data
    print("Loading data")
    fn_gaia = '../data/gaia_clean.fits'
    tab_gaia = utils.load_table(fn_gaia)
    # TEST ONLY W SMALL AMOUNT
    #tab_gaia = tab_gaia[np.random.randint(0, len(tab_gaia), size=10000)]
    N_gaia = len(tab_gaia)
    print(f"N of clean gaia catalog: {N_gaia}")

    # Make Gmax cut, because will be more robust that way if we cut first
    i_makeGcut = tab_gaia['phot_g_mean_mag'] < G_max
    tab_gaia = tab_gaia[i_makeGcut]
    print(f"N after G_max cut of G<{G_max}:", len(tab_gaia))

    # Construct full feature matrix
    print("Constructing feature matrix")
    #feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2', 'phot_g_mean_mag']
    #feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2']
    feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2',
                    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'w1mpro', 'w2mpro']
    if fn_prev_estimate is not None:
        print(f"Including SPZ from prev estimate, {fn_prev_estimate}")
        tab_spz_prev = Table.read(fn_prev_estimate, format='fits')
        tab_spz_prev.keep_columns(['source_id', 'redshift_spzqsoc'])
        tab_gaia = astropy.table.join(tab_gaia, tab_spz_prev, keys='source_id', join_type='left')
        feature_keys.append('redshift_spzqsoc')

    X_gaia = construct_X(tab_gaia, feature_keys)
    print('prev spzs')
    print(X_gaia[:,-1])
    Y_qsoc = tab_gaia['redshift_qsoc']
    print("N after throwing out bad features:", len(tab_gaia))
    i_has_sdss_redshift = np.isfinite(tab_gaia['sdss_Z'])
    print("N with SDSS redshifts:", np.sum(i_has_sdss_redshift))

    # Split training (where have SDSS redshifts) and not
    X_labeled = X_gaia[i_has_sdss_redshift]
    Y_labeled = tab_gaia[i_has_sdss_redshift]['sdss_Z']
    Y_qsoc_labeled = Y_qsoc[i_has_sdss_redshift]
    N_labeled = X_labeled.shape[0]

    if mode=='2step':
        dz_thresh = 0.1
        dz = np.abs(np.array(Y_qsoc_labeled) - np.array(Y_labeled))/(1 + np.array(Y_labeled))
        # this will be 1 where dz is above the threshold (correct z), 0 if incorrect
        i_correctqsoc = dz<dz_thresh
        C_class = np.full(len(dz), 0.0)
        C_class[i_correctqsoc] = 1.0
        print(f"Fraction QSOC within |dz|<{dz_thresh} [already correct]: {np.sum(i_correctqsoc)/len(i_correctqsoc):.3f}")
        print(f"Fraction QSOC with |dz|>{dz_thresh} [incorrect]: {np.sum(~i_correctqsoc)/len(i_correctqsoc):.3f}")

    elif mode=='zbins':
        #z_bins = np.percentile(Y_labeled, np.linspace(0, 100, N_classes+1))
        #z_bins = np.array([0.0, 1.15, 1.4, 1.7, 2, 2.3, 8])
        z_bins = np.array([0.0, 1.15, 2.3, 8])
        assert np.min(Y_labeled)>0, "spectroscopic zs should all be positive!"
        # widen first and last bins to make sure min and max zs fit in
        z_bins[0] = 0
        z_bins[-1] += 0.1
        print("z bins:", list(z_bins))
        # minus 1 because bins returns a value 1-N_bins, we want to start with 0 for class labeling 
        C_class = np.digitize(Y_labeled, z_bins)-1
        #i_highz = Y_labeled > z_thresh
        #C_class = np.full(len(i_highz), 0.0)
        #C_class[i_highz] = 1.0
        for n in range(N_classes):
            print(f"Fraction with C={n}, {z_bins[n]:.2f}<z<{z_bins[n+1]:.2f}: {np.sum(C_class==n)/len(C_class):.3f}")

    # this is essentially shuffling an array of 1-N
    #random_ints = rng.choice(range(N_labeled), size=N_labeled, replace=False)
    rand_ints_labeled = tab_gaia['rand_ints_clean'][i_has_sdss_redshift]
    # N_tot=N_gaia because the numbers go up to all the ones in the clean catalog
    idx_train, idx_valid, idx_test = utils.split_train_val_test(rand_ints_labeled, N_tot=N_gaia)
    chunk_spz = np.empty(N_labeled, dtype='S5')
    chunk_spz[idx_train] = 'train'
    chunk_spz[idx_valid] = 'valid'
    chunk_spz[idx_test] = 'test'

    # split into actual training and validation subset
    #int_divider = int(frac_test*N_gaia)
    # test are ones that are not used to validate the model but have labels to do final check
    # idxs_test = np.where(rand_ints_labeled < int_divider)[0]
    # idxs_train = np.where(rand_ints_labeled >= int_divider)[0]
    
    print(f"N_train: {len(idx_train)} ({len(idx_train)/N_labeled:.3f})") 
    print(f"N_valid: {len(idx_valid)} ({len(idx_valid)/N_labeled:.3f})") 
    print(f"N_test: {len(idx_test)} ({len(idx_test)/N_labeled:.3f})") 

    X_train = X_labeled[idx_train]
    X_valid = X_labeled[idx_valid]
    X_test = X_labeled[idx_test]

    Y_train = Y_labeled[idx_train]
    Y_valid = Y_labeled[idx_valid]
    Y_test = Y_labeled[idx_test]

    if mode=='2step' or mode=='zbins':
        C_train = C_class[idx_train]
        C_valid = C_class[idx_valid]
        C_test = C_class[idx_test]

    # Apply to all, including those with SDSS redshifts (for consistency)
    X_apply = X_gaia
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_apply: {X_apply.shape}")
    assert X_train.shape[0]==Y_train.shape[0], "X and Y must have same length!"

    # Run redshift estimation
    print("Running redshift estimation")
    # cross_validate(redshift_estimator_class, 
    #                X_train, Y_train,
    #                redshift_estimator_kwargs, rng)
    if mode=='2step':            

        redshift_estimator_class_classifier = \
                redshift_estimator_dict[redshift_estimator_name_classifier]                        
        redshift_estimator_kwargs_classifier = redshift_estimator_kwargs_dict[redshift_estimator_name_classifier]

        # just X_test for now, but will want X_apply
        if os.path.exists(fn_model_classifier) and not overwrite_classifier:
            print(f"Classifier {fn_model_classifier} already exists and overwrite_classifier={overwrite_classifier}")
            print("So load it in!")
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_apply=X_test, 
                                                                                train_mode=False, test_mode=True)
            redshift_estimator_classifier.load_model(fn_model_classifier)
        else:           
            print(f"Training classifier {fn_model_classifier} (overwrite_classifier={overwrite_classifier})")                                                             
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_train, C_train, X_valid, C_valid, X_test,
                                                                                **redshift_estimator_kwargs_classifier)
            redshift_estimator_classifier.train()
            # save
            redshift_estimator_classifier.save_model(fn_model_classifier)

        # Apply classifier
        redshift_estimator_classifier.apply()  

        # Get classifications
        #C_pred_test = redshift_estimator_classifier.Y_hat_apply[i_has_sdss_redshift][idx_test]
        prob_pred_test = redshift_estimator_classifier.Y_hat_apply
        class_thresh = 0.5
        i_pred_test_correctqsoc = prob_pred_test > class_thresh
        C_pred_test = np.full(len(prob_pred_test), 0.0)
        C_pred_test[i_pred_test_correctqsoc] = 1.0
        print(f"Fraction accurate predicted labels, test set: {np.sum(C_pred_test==C_test)/len(C_test):.3f}")
        print(f"Fraction of test QSOCs that are predicted to be correct: {np.sum(i_pred_test_correctqsoc)/len(i_pred_test_correctqsoc):.3f}")

        # Now train on only incorrect ones (which we know for sure)
        # need indices in this form to combine with i_correctqsoc
        i_train = np.full(len(X_labeled), False)
        i_train[idx_train] = True
        i_valid = np.full(len(X_labeled), False)
        i_valid[idx_valid] = True
        i_test = np.full(len(X_labeled), False)
        i_test[idx_test] = True

        X_train_incorrectqsoc = X_labeled[i_train & ~i_correctqsoc]
        Y_train_incorrectqsoc = Y_labeled[i_train & ~i_correctqsoc]

        X_valid_incorrectqsoc = X_labeled[i_valid & ~i_correctqsoc]
        Y_valid_incorrectqsoc = Y_labeled[i_valid & ~i_correctqsoc]
        print(X_train_incorrectqsoc.shape, X_valid_incorrectqsoc.shape)
       
        # For test sources that classifier says are correct, just leave QSOC redshift! 
        # For test sources that classifier says are wrong, apply regressor NN
        # test for now, but should be apply
        # Y_hat_apply = np.empty(X_apply.shape[0])
        # Y_hat_apply[i_pred_apply_correctqsoc] = Y_qsoc[i_pred_apply_correctqsoc]
        # X_apply_incorrectqsoc = X_apply[~i_pred_apply_correctqsoc]
        Y_hat_test = np.empty(X_test.shape[0])
        Y_hat_test[i_pred_test_correctqsoc] = Y_qsoc_labeled[idx_test][i_pred_test_correctqsoc]
        X_test_incorrectqsoc = X_test[~i_pred_test_correctqsoc]

        if os.path.exists(fn_model) and not overwrite_model:
            print(f"Model {fn_model} already exists and overwrite_model={overwrite_model}")
            print("So load it in!")
            redshift_estimator = redshift_estimator_class(X_apply=X_test_incorrectqsoc, 
                                                          train_mode=False, test_mode=True)
            redshift_estimator.load_model(fn_model)
        else: 
            print(f"Training model {fn_model} (overwrite_model={overwrite_model})")        
            # redshift_estimator = redshift_estimator_class(X_train, Y_train, 
            #                                             X_valid, Y_valid, 
            #                                             X_test_incorrectqsoc, **redshift_estimator_kwargs)
            redshift_estimator = redshift_estimator_class(X_train_incorrectqsoc, Y_train_incorrectqsoc, 
                                                          X_valid_incorrectqsoc, Y_valid_incorrectqsoc, 
                                                          X_test_incorrectqsoc, **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.save_model(fn_model)

        # Apply
        redshift_estimator.apply()

        Y_hat_test[~i_pred_test_correctqsoc] = redshift_estimator.Y_hat_apply
        print(X_test.shape)
        print(Y_hat_test.shape)

        # Print results
        dzs = [0.01, 0.1, 0.2, 1.0]
        for dz in dzs:
            print(f"Fraction recovered with Dz/(1+z)<{dz}:")
            Y_test_correctqsoc = Y_labeled[i_test & i_correctqsoc]
            frac_recovered = utils.get_fraction_recovered(Y_test_correctqsoc, 
                                                          Y_hat_test[i_correctqsoc[idx_test]], dz)
            print(f"When QSOC is 'correct' (cheating to use), frac actually correct: {frac_recovered:.3f}")            
            frac_recovered = utils.get_fraction_recovered(Y_test[i_pred_test_correctqsoc], 
                                                          Y_hat_test[i_pred_test_correctqsoc], dz)
            print(f"QSOCs predicted to be correct: {frac_recovered:.3f}")
            frac_recovered = utils.get_fraction_recovered(Y_test[~i_pred_test_correctqsoc], 
                                                          Y_hat_test[~i_pred_test_correctqsoc], dz)
            print(f"QSOCs orig predicted to be incorrect: {frac_recovered:.3f}")
            frac_recovered = utils.get_fraction_recovered(Y_test, Y_hat_test, dz)
            print(f"SPZ: {frac_recovered:.3f}")
            frac_recovered_qsoc = utils.get_fraction_recovered(Y_test, Y_qsoc_labeled[idx_test], dz)
            print(f"QSOC: {frac_recovered_qsoc:.3f}")

    elif mode=='zbins':

        redshift_estimator_class_classifier = \
                redshift_estimator_dict[redshift_estimator_name_classifier]                        
        redshift_estimator_kwargs_classifier = redshift_estimator_kwargs_dict[redshift_estimator_name_classifier]

        # just X_test for now, but will want X_apply
        if os.path.exists(fn_model_classifier) and not overwrite_classifier:
            print(f"Classifier {fn_model_classifier} already exists and overwrite_classifier={overwrite_classifier}")
            print("So load it in!")
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_apply=X_test, 
                                                                                train_mode=False, test_mode=True)
            redshift_estimator_classifier.load_model(fn_model_classifier)
        else:           
            print(f"Training classifier {fn_model_classifier} (overwrite_classifier={overwrite_classifier})")                                                             
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_train, C_train, X_valid, C_valid, X_test,
                                                                                **redshift_estimator_kwargs_classifier)
            redshift_estimator_classifier.train()
            # save
            redshift_estimator_classifier.save_model(fn_model_classifier)

        # Apply classifier
        redshift_estimator_classifier.apply()  

        # Get classifications
        C_pred_test = redshift_estimator_classifier.Y_hat_apply
        print(f"Fraction accurate predicted labels, test set: {np.sum(C_pred_test==C_test)/len(C_test):.3f}")
        for n in range(N_classes):
            print(f"Fraction accurate for true C={n}: {np.sum(C_pred_test[C_test==n]==C_test[C_test==n])/len(C_test[C_test==n]):.3f}")
            print(f"Fraction accurate for pred C={n}: {np.sum(C_pred_test[C_pred_test==n]==C_test[C_pred_test==n])/len(C_test[C_pred_test==n]):.3f}")

        #print(f"Fraction of test that are predicted to be high-z: {np.sum(i_pred_test_highz)/len(i_pred_test_highz):.3f}")

        # Now train on only incorrect ones (which we know for sure)
        # need indices in this form to combine with i_correctqsoc

        Y_hat_test = np.empty(X_test.shape[0])
        for n in range(N_classes):
            X_train_zsub = X_train[C_train==n]
            Y_train_zsub = Y_train[C_train==n]
            X_valid_zsub = X_valid[C_valid==n]
            Y_valid_zsub = Y_valid[C_valid==n]

            X_test_zsub_pred = X_test[C_pred_test==n]

            redshift_estimator = redshift_estimator_class(X_train_zsub, Y_train_zsub, 
                                                          X_valid_zsub, Y_valid_zsub, 
                                                          X_test_zsub_pred, 
                                                          **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.apply()

            Y_hat_test[C_pred_test==n] = redshift_estimator.Y_hat_apply

        # Print results
        dzs = [0.01, 0.1, 0.2, 1.0]
        for dz in dzs:
            print(f"Fraction recovered with Dz/(1+z)<{dz}:")
            frac_recovered = utils.get_fraction_recovered(Y_test, Y_hat_test, dz)
            print(f"SPZ: {frac_recovered:.3f}")
            frac_recovered_qsoc = utils.get_fraction_recovered(Y_test, Y_qsoc_labeled[idx_test], dz)
            print(f"QSOC: {frac_recovered_qsoc:.3f}")
            for n in range(N_classes):
                frac_recovered_zsub = utils.get_fraction_recovered(Y_test[C_test==n], 
                                                                   Y_hat_test[C_test==n], dz)
                print(f"    SPZ, true C={n}: {frac_recovered_zsub:.3f}")
                frac_recovered_pred = utils.get_fraction_recovered(Y_test[C_pred_test==n], 
                                                                   Y_hat_test[C_pred_test==n], dz)
                print(f"    SPZ, pred C={n}: {frac_recovered_pred:.3f}")

    elif mode=='outliers':            

        if os.path.exists(fn_model) and not overwrite_model:
            print(f"Model {fn_model} already exists and overwrite_model={overwrite_model}")
            print("So load it in!")
            redshift_estimator = redshift_estimator_class(X_apply=X_test, 
                                                          train_mode=False, test_mode=True)
            redshift_estimator.load_model(fn_model)
        else: 
            print(f"Training model {fn_model} (overwrite_model={overwrite_model})")        
            redshift_estimator = redshift_estimator_class(X_train, Y_train, 
                                                        X_valid, Y_valid, 
                                                        X_labeled, **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.save_model(fn_model)
            
        # Apply
        redshift_estimator.apply()
        # TODO: really should do this on an independent, second training set i think (e.g. validation)
        Y_hat_labeled_initial = redshift_estimator.Y_hat_apply

        # Now setup classifier to detect those that are still outliers
        dz_thresh = 0.2
        dz = np.abs(np.array(Y_hat_labeled_initial) - np.array(Y_labeled))/(1 + np.array(Y_labeled))
        #dz_noabs = (np.array(Y_hat_labeled_initial) - np.array(Y_labeled))/(1 + np.array(Y_labeled))

        # this will be 1 where dz is above the threshold (correct z), 0 if incorrect
        i_correctinitial = dz<dz_thresh
        #print("classify one outlier line")
        #i_correctinitial = (dz_noabs>0.3) | (dz_noabs<0.2)
        C_class = np.full(len(dz), 0.0)
        C_class[i_correctinitial] = 1.0
        print(f"Fraction initial SPZ within |dz|<{dz_thresh} [correct]: {np.sum(i_correctinitial)/len(i_correctinitial):.3f}")
        print(f"Fraction initial SPZ with |dz|>{dz_thresh} [incorrect]: {np.sum(~i_correctinitial)/len(i_correctinitial):.3f}")

        C_train = C_class[idx_train]
        C_valid = C_class[idx_valid]
        C_test = C_class[idx_test]

        # Replace QSOC in features with SPZ because SPZ better!
        X_train[:,0] = Y_hat_labeled_initial[idx_train]
        X_valid[:,0] = Y_hat_labeled_initial[idx_valid]
        X_test[:,0] = Y_hat_labeled_initial[idx_test]

        redshift_estimator_class_classifier = \
                redshift_estimator_dict[redshift_estimator_name_classifier]                        
        redshift_estimator_kwargs_classifier = redshift_estimator_kwargs_dict[redshift_estimator_name_classifier]

        # just X_test for now, but will want X_apply
        if os.path.exists(fn_model_classifier) and not overwrite_classifier:
            print(f"Classifier {fn_model_classifier} already exists and overwrite_classifier={overwrite_classifier}")
            print("So load it in!")
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_apply=X_test, 
                                                                                train_mode=False, test_mode=True)
            redshift_estimator_classifier.load_model(fn_model_classifier)
        else:           
            print(f"Training classifier {fn_model_classifier} (overwrite_classifier={overwrite_classifier})")                                                             
            redshift_estimator_classifier = redshift_estimator_class_classifier(X_train, C_train, X_valid, C_valid, X_test,
                                                                                **redshift_estimator_kwargs_classifier)
            redshift_estimator_classifier.train()
            # save
            redshift_estimator_classifier.save_model(fn_model_classifier)

        # Apply classifier
        redshift_estimator_classifier.apply()  

        # Get classifications
        #C_pred_test = redshift_estimator_classifier.Y_hat_apply[i_has_sdss_redshift][idx_test]
        if redshift_estimator_class_classifier=='SVM':
            C_pred_test = redshift_estimator_classifier.Y_hat_apply
        else:    
            prob_pred_test = redshift_estimator_classifier.Y_hat_apply
            # make low so it has to be pretty sure its incorrect
            class_thresh = 0.3
            i_pred_test_correctinitial = prob_pred_test > class_thresh
            C_pred_test = np.full(len(prob_pred_test), 0.0)
            C_pred_test[i_pred_test_correctinitial] = 1.0
        print(f"N actually correct: {np.sum(C_test==1)}")
        print(f"N actually incorrect: {np.sum(C_test==0)}")
        print(f"Fraction accurate predicted labels, test set: {np.sum(C_pred_test==C_test)/len(C_test):.3f}")
        print(f"Fraction of initial SPZs that are predicted to be correct: {np.sum(i_pred_test_correctinitial)/len(i_pred_test_correctinitial):.3f}")
        print(f"    Fraction of correct initial SPZs that are predicted to be correct: {np.sum(C_pred_test[C_test==1]==1)/len(C_test[C_test==1]):.3f} (N={np.sum(C_pred_test[C_test==1]==1)})")
        print(f"    Fraction of incorrect initial SPZs that are predicted to be incorrect: {np.sum(C_pred_test[C_test==0]==0)/len(C_test[C_test==0]):.3f} (N={np.sum(C_pred_test[C_test==0]==0)})")
        print(f"    Fraction of correct initial SPZs that are predicted to be incorrect: {np.sum(C_pred_test[C_test==1]==0)/len(C_test[C_test==1]):.3f} (N={np.sum(C_pred_test[C_test==1]==0)})")
        print(f"    Fraction of incorrect initial SPZs that are predicted to be correct: {np.sum(C_pred_test[C_test==0]==1)/len(C_test[C_test==0]):.3f} (N={np.sum(C_pred_test[C_test==0]==1)})")
        #print(pausehere)

        # Now train on only incorrect ones (which we know for sure)
        X_train_incorrectinitial = X_train[C_train==0]
        Y_train_incorrectinitial = Y_train[C_train==0]
        # X_valid_incorrectinitial = X_valid[C_valid==0]
        # Y_valid_incorrectinitial = Y_valid[C_valid==0]
        print(X_train_incorrectinitial.shape, Y_train_incorrectinitial.shape)
       
        # Y_hat_apply = np.empty(X_apply.shape[0])
        # Y_hat_apply[i_pred_apply_correctqsoc] = Y_qsoc[i_pred_apply_correctqsoc]
        # X_apply_incorrectqsoc = X_apply[~i_pred_apply_correctqsoc]
        # if we predicted that the initial was right, keep it. else, apply outlier-only kNN
        Y_hat_test = np.empty(X_test.shape[0])
        Y_hat_test[i_pred_test_correctinitial] = Y_hat_labeled_initial[idx_test][i_pred_test_correctinitial]
        X_test_incorrectinitial = X_test[~i_pred_test_correctinitial]
        print(X_test_incorrectinitial.shape)

        knn_mode = 'zbins'
        if knn_mode=='single_bin':
            # just knn so dont worry about saving and loading
            assert redshift_estimator_name=='kNN', "Only use kNN for this outlier step for now!"
            redshift_estimator = redshift_estimator_class(X_train=X_train_incorrectinitial, 
                                                        Y_train=Y_train_incorrectinitial, 
                                                        X_apply=X_test_incorrectinitial, **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.apply()
            Y_hat_test[~i_pred_test_correctinitial] = redshift_estimator.Y_hat_apply
        elif knn_mode=='zbins':
            z_bins = np.array([0.0, 1.3, 2.3, 8.0])
            N_zbins = len(z_bins)-1
            C_zbin = np.digitize(Y_hat_labeled_initial, z_bins)-1
            for n in range(N_zbins):
                # in zbin and incorrect initial guess
                i_zbin_train = (C_zbin[idx_train]==n) & (C_train==0)
                i_zbin_test = (C_zbin[idx_test]==n) & (C_pred_test==0)
                print(f"N_train in {z_bins[n]:.2f}<z<{z_bins[n+1]:.2f}: {np.sum(i_zbin_train)}")
                print(f"N_test in {z_bins[n]:.2f}<z<{z_bins[n+1]:.2f}: {np.sum(i_zbin_test)}")
                redshift_estimator = redshift_estimator_class(X_train=X_train[i_zbin_train], 
                                                            Y_train=Y_train[i_zbin_train], 
                                                            X_apply=X_test[i_zbin_test], **redshift_estimator_kwargs)
                redshift_estimator.train()
                redshift_estimator.apply()
                Y_hat_test[i_zbin_test] = redshift_estimator.Y_hat_apply

        # Print results
        dzs = [0.01, 0.1, 0.2, 1.0]
        for dz in dzs:
            print(f"Fraction recovered with Dz/(1+z)<{dz}:")
            frac_recovered = utils.get_fraction_recovered(Y_test, Y_hat_test, dz)
            print(f"SPZ: {frac_recovered:.3f}")
            frac_recovered_qsoc = utils.get_fraction_recovered(Y_test, Y_qsoc_labeled[idx_test], dz)
            print(f"QSOC: {frac_recovered_qsoc:.3f}")

    else:
        if os.path.exists(fn_model) and not overwrite_model:
            print(f"Model {fn_model} already exists and overwrite_model={overwrite_model}")
            print("So load it in!")
            redshift_estimator = redshift_estimator_class(X_apply=X_test, 
                                                          train_mode=False, test_mode=True)
            redshift_estimator.load_model(fn_model)
        else:
            print(f"Training model {fn_model} (overwrite_model={overwrite_model})")                                                             
            redshift_estimator = redshift_estimator_class(X_train, Y_train, X_valid, Y_valid, X_test,
                                                          **redshift_estimator_kwargs)
            # for now lets just apply to validation set, deal w full thing later
            #redshift_estimator = redshift_estimator_class(X_train, Y_train, X_test, **redshift_estimator_kwargs)
            redshift_estimator.train()
            redshift_estimator.save_model(fn_model)

        # Apply
        redshift_estimator.apply()
        #Y_hat_apply = redshift_estimator.Y_hat_apply
        Y_hat_test = redshift_estimator.Y_hat_apply

        Y_qsoc_test = Y_qsoc_labeled[idx_test]

        # if spz and qsoc agree, assign qsoc    
        dz_spzqsoc_test = np.abs(np.array(Y_hat_test) - np.array(Y_qsoc_test))/(1 + np.array(Y_qsoc_test))
        # this will be 1 where dz is above the threshold (correct z), 0 if incorrect
        dz_thresh = 0.1
        i_agree_spzqsoc_test = dz_spzqsoc_test<dz_thresh
        Y_hat_test_spzqsoc = Y_hat_test.copy()
        Y_hat_test_spzqsoc[i_agree_spzqsoc_test] = Y_qsoc_test[i_agree_spzqsoc_test]

        # Print results
        dzs = [0.01, 0.1, 0.2, 1.0]
        for dz in dzs:
            print(f"Fraction recovered with Dz/(1+z)<{dz}:")
            frac_recovered = utils.get_fraction_recovered(Y_test, Y_hat_test, dz)
            print(f"SPZ: {frac_recovered:.3f}")
            frac_recovered_spzqsoc = utils.get_fraction_recovered(Y_test, Y_hat_test_spzqsoc, dz)
            print(f"SPZ-QSOC: {frac_recovered_spzqsoc:.3f}")
            frac_recovered_qsoc = utils.get_fraction_recovered(Y_test, Y_qsoc_test, dz)
            print(f"QSOC: {frac_recovered_qsoc:.3f}")

    print(slfsfj)
    # Print results
    dzs = [0.01, 0.1, 0.2, 1.0]
    for dz in dzs:
        print(f"Fraction recovered with Dz/(1+z)<{dz}:")
        frac_recovered = utils.get_fraction_recovered(Y_test, Y_hat_apply[i_has_sdss_redshift][idx_test], dz)
        print(f"SPZ: {frac_recovered:.3f}")
        frac_recovered_qsoc = utils.get_fraction_recovered(Y_test, Y_qsoc_labeled[idx_test], dz)
        print(f"QSOC: {frac_recovered_qsoc:.3f}")

    # # Save model
    # redshift_estimator.save_model(fn_model)

    # Save results
    print("Save results")
    columns_to_keep = ['source_id', 'sdss_OBJID', 'phot_g_mean_mag', 'redshift_qsoc', 'sdss_Z', 'rand_ints_clean']
    tab_gaia.keep_columns(columns_to_keep)
    tab_gaia['redshift_spz'] = Y_hat_apply
    tab_gaia['redshift_spz_err'] = redshift_estimator.sigma_z
    tab_gaia['chunk_spz'] = np.full(len(tab_gaia), '', dtype='S5')
    tab_gaia['chunk_spz'][i_has_sdss_redshift] = chunk_spz
    tab_gaia.write(fn_spz, overwrite_table=overwrite_table)
    print(f"Wrote specphotozs to {fn_spz}!")


def construct_X(tab, feature_keys):

    X = []
    for feature_key in feature_keys:
        X.append(tab[feature_key])
    X = np.array(X).T
    i_badfeat = np.any(~np.isfinite(X), axis=1)
    # shouldn't be any bad features because we cleaned up first in make_data_tables.gaia_clean()
    assert np.sum(i_badfeat)==0, "Some bad feature data in clean catalog!"
    
    return X



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
                       train_mode=True, test_mode=False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_apply = X_apply
        self.train_mode = train_mode
        self.test_mode = test_mode


    def train(self):
        pass 


    def predict(self):
        pass


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
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)


    def train(self):
        print("Training")
        self.tree_train = self.build_tree(self.X_train_scaled)


    def apply(self):
        print("Applying")
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def build_tree(self, X):
        print("Building kdTree")
        return KDTree(X)


    def predict(self, X_input_scaled):
        print("Getting median Z of nearest neighbors")
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
        low_z, Y_hat, up_z = np.percentile(self.Y_train[inds_nodist0], (2.5, 50, 97.5), axis=1)
        sigma_z = (up_z - low_z)/4
        return Y_hat, sigma_z


class RedshiftEstimatorSVM(RedshiftEstimator):

    
    def __init__(self, *args, C=1.0, gamma=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = C
        self.gamma = gamma
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
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)


    def train(self):
        print("Training")
        #$self.model = sklearn.svm.SVC(C=self.C, gamma=self.gamma, kernel="rbf", probability=False)
        print(self.C)
        #self.model = sklearn.svm.LinearSVC(C=self.C) 
        self.model = sklearn.svm.LinearSVC(class_weight='balanced')
        self.model.fit(self.X_train, self.Y_train)

    def apply(self):
        print("Applying")
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def predict(self, X_input_scaled):
        Y_hat = self.model.predict(X_input_scaled)
        sigma_z = [np.NaN]*len(Y_hat) 
        return Y_hat, sigma_z



class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size=32, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.SELU()
        #self.do1 = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act2 = torch.nn.SELU()
        #self.do2 = torch.nn.Dropout(0.2)
        self.lin3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act3 = torch.nn.SELU()
        #self.do3 = torch.nn.Dropout(0.2)
        self.linfinal = torch.nn.Linear(self.hidden_size, output_size)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        torch.nn.init.zeros_(self.lin3.bias)
        torch.nn.init.xavier_uniform_(self.linfinal.weight)
        torch.nn.init.zeros_(self.linfinal.bias)
        self.double()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        #x = self.do1(x)
        x = self.lin2(x)
        x = self.act2(x)
        #x = self.do2(x)
        x = self.lin3(x)
        x = self.act3(x)
        #x = self.do3(x)
        output = self.linfinal(x)
        return output


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    print('worker seed', worker_seed)


class RedshiftEstimatorANN(RedshiftEstimator):
    
    def __init__(self, *args, rng=None, learning_rate=0.005, batch_size=512, **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self, frac_valid=0.2):
        # switched to passing in validation data
        # N_train = self.X_train.shape[0]
        # # assign unique ints to the training set
        # random_ints = self.rng.choice(range(N_train), size=N_train, replace=False)
        # # split into actual training and validation subset
        # int_divider = int(frac_valid*N_train)
        # idx_valid = np.where(random_ints < int_divider)[0]
        # idx_train = np.where(random_ints >= int_divider)[0]
        # print("N_train:", len(idx_train), "N_valid:", len(idx_valid))

        # self.X_train_sub = self.X_train[idx_train]
        # self.Y_train_sub = self.Y_train[idx_train]
        # self.X_valid = self.X_train[idx_valid]
        # self.Y_valid = self.Y_train[idx_valid]

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        self.scale_y()

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub_scaled)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid_scaled)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)
        # print(self.X_train_sub[0])
        # print(self.X_train_sub_scaled[0])


    def scale_y(self):
        self.scaler_y = StandardScaler(with_mean=False, with_std=False)
        self.scaler_y.fit(np.atleast_2d(self.Y_train_sub).T)
        self.Y_train_sub_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_train_sub).T)
        self.Y_valid_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_valid).T)
        # print(self.Y_train_sub[:5])
        # print(self.Y_train_sub_scaled[:5])


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid



    def train(self, hidden_size=512, max_epochs=20, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = 1 # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.GaussianNLLLoss()
        def loss_dz(output, target):
            loss = torch.mean(torch.divide(torch.abs(output - target), torch.add(target, 1.0)))
            return loss
        #self.criterion = loss_dz
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(torch.from_numpy(X_input_scaled).double())

        y_pred_scaled = y_pred_scaled.squeeze().numpy()
        print(y_pred_scaled.shape)
        y_pred = np.squeeze(self.scaler_y.inverse_transform(np.atleast_2d(y_pred_scaled).T))
        #y_pred = y_pred_scaled
        print(y_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return y_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'scaler_y': self.scaler_y,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        self.scaler_y = model_checkpoint['scaler_y']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']



class RedshiftEstimatorANN2class(RedshiftEstimator):
    
    def __init__(self, *args, rng=None, learning_rate=0.005, batch_size=512, **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self):

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        print(self.Y_train_sub)

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid



    def train(self, hidden_size=512, max_epochs=20, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = 1 # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        # binary cross entropy including a sigmoid to squeeze output of NN into 0-1
        # pos_weight: https://discuss.pytorch.org/t/bcewithlogitsloss-calculating-pos-weight/146336/3
        N_pos = np.sum(self.Y_train)
        N = len(self.Y_train)
        pos_weight = N/N_pos - 1 #this is equiv to N_neg/N_pos, as that forum says
        
        print('pos_weight:', N_pos, N, pos_weight)
        pos_weight *= 0.7
        print("DEFLATING POS_WEIGHT, now", pos_weight)

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(X_input_scaled).double())

        # to get from logits to probabilities
        # https://sebastianraschka.com/blog/2022/losses-learned-part1.html
        c_pred = torch.sigmoid(y_pred)
        c_pred = c_pred.squeeze().numpy()
        print(c_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return c_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']


class RedshiftEstimatorANNmulticlass(RedshiftEstimator):
    
    def __init__(self, *args, rng=None, learning_rate=0.005, batch_size=512, 
                 N_classes=1, **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.N_classes = N_classes
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self):

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        print(self.Y_train_sub)

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply_raw, self.sigma_z = self.predict(self.X_apply_scaled)
        ### _,pred = torch.max(out, dim=1)
        # the raw data is the probabilities; argmax to get highest-prob zbin
        self.Y_hat_apply = self.Y_hat_apply_raw.argmax(axis=1)
        print(self.Y_hat_apply_raw[0])
        print(self.Y_hat_apply[0])
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid


    def train(self, hidden_size=512, max_epochs=20, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = self.N_classes # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        # binary cross entropy including a sigmoid to squeeze output of NN into 0-1
        # pos_weight: https://discuss.pytorch.org/t/bcewithlogitsloss-calculating-pos-weight/146336/3
        N = len(self.Y_train)
        weights = [N/np.sum(self.Y_train==i)-1 for i in range(self.N_classes)]
        print('frac in each class:', [np.sum(self.Y_train==i)/N for i in range(self.N_classes)])
        print('weights', weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(X_input_scaled).double())

        # to get from logits to probabilities
        # https://sebastianraschka.com/blog/2022/losses-learned-part1.html
        c_pred = torch.sigmoid(y_pred)
        c_pred = c_pred.squeeze().numpy()
        print(c_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return c_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']


class DataSet(Dataset):

    def __init__(self, X, Y, y_var=None, randomize=True):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.y_var = y_var
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")
        if y_var is not None:
            self.y_var = np.array(self.y_var)
            if len(self.X) != len(self.y_var):
                raise Exception("The length of X does not match the length of y_var")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        if self.y_var is not None:
            _y_var = self.y_var[index]
            return _x, _y, _y_var
        return _x, _y




if __name__=='__main__':
    main()