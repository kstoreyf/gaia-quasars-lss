import numpy as np
import os
from numpy.random import default_rng

import utils


def main():
    # Full sample to apply to
    fn_gwnec = '../data/gaia_candidates_wnec.fits'
    tab_gwnec = utils.load_table(fn_gwnec, format='fits')
    print(f"Number of Gaia quasar candidates with necessary data: {len(tab_gwnec)}")

    # Labeled sample
    fn_labeled = '../data/labeled_wnec.fits'
    tab_labeled = utils.load_table(fn_labeled)
    print(f"Number of labeled Gaia quasar candidates for training/validation: {len(tab_labeled)}")

    X_labeled = construct_X(tab_labeled)
    i_train, i_valid, i_test = utils.split_train_val_test(tab_labeled['rand_ints'], frac_train=0.5, frac_test=0.25, frac_val=0.25)
    
    X_train = X_labeled[i_train]
    X_valid = X_labeled[i_valid]

    y_labeled = tab_labeled['class']
    y_labeled[y_labeled=='s'] = 'o'
    y_labeled[y_labeled=='g'] = 'o'

    y_train = y_labeled[i_train]
    y_valid = y_labeled[i_valid]

    y_valid_pred = two_lines(X_train, y_train, X_valid)

    print("Baseline: take all to be quasars")
    y_valid_baseline = ['q']*len(y_valid)
    compute_metrics(y_valid_baseline, y_valid)

    print("Results")
    compute_metrics(y_valid_pred, y_valid)


def compute_metrics(y_pred, y_true):
    class_labels = list(set(y_true))
    conf_mat = utils.confusion_matrix(y_pred, y_true, class_labels)
    print(conf_mat)
    comp = utils.completeness(conf_mat, class_labels, label='q')
    fp = utils.N_FP(conf_mat, class_labels, label='q')
    purity = utils.purity(conf_mat, class_labels, label='q')
    print("Completeness:", comp)
    print("Number of false positives:", fp)
    print("Purity:", purity)



def construct_X(tab):
    X = np.array([tab['phot_g_mean_mag']-tab['mag_w1_vg'],
                  tab['mag_w1_vg']-tab['mag_w2_vg']]).T
    return X


# decontamination models

# no training  required but pass for consistency
def two_lines(X_train, y_train, X_valid):
    cut_0 = 0., 1., 0.2
    cut_1 = 1., 1., 2.9
    color_cuts = [cut_0, cut_1]

    # this predicts that all that make cuts are quasars, others are contaminants
    idx_predq = utils.gw1_w1w2_cuts_index(X_valid[:,0], X_valid[:,1], color_cuts)
    y_pred = np.full(X_valid.shape[0], 'o')
    y_pred[idx_predq] = 'q'

    return y_pred


def svm(X_train, y_train, X_valid):
    from sklearn import svm
    clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=0.0001)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    return y_pred


if __name__=='__main__':
    main()