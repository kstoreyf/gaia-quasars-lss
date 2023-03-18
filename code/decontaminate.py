import numpy as np
import os
from numpy.random import default_rng

from scipy.optimize import minimize

import utils


def main():

    # set params
    model = two_lines
    #model = svm
    fn_model = '../data/decontamination_models/model_2lines_optimize.npy'

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

    class_labels = ['q', 'o']
    y_labeled = tab_labeled['class']
    y_labeled[y_labeled=='s'] = 'o'
    y_labeled[y_labeled=='g'] = 'o'

    y_train = y_labeled[i_train]
    y_valid = y_labeled[i_valid]

    y_valid_pred = model(X_train, y_train, X_valid, class_labels=class_labels, fn_model=fn_model)

    print("Baseline: take all to be quasars")
    y_valid_baseline = ['q']*len(y_valid)
    compute_metrics(y_valid_baseline, y_valid, class_labels)

    print("Results")
    compute_metrics(y_valid_pred, y_valid, class_labels)


def compute_metrics(y_pred, y_true, class_labels):
    conf_mat = utils.confusion_matrix(y_pred, y_true, class_labels)
    print(conf_mat)
    comp = utils.completeness(conf_mat, class_labels, label='q')
    fn = utils.N_FN(conf_mat, class_labels, label='q')
    fp = utils.N_FP(conf_mat, class_labels, label='q')
    purity = utils.purity(conf_mat, class_labels, label='q')
    print(f"Completeness: {comp:.4f}")
    print("Number of false negatives:", fn)
    print(f"Purity: {purity:.4f}")
    print("Number of false positives:", fp)



def construct_X(tab):
    X = np.array([tab['phot_g_mean_mag']-tab['mag_w1_vg'],
                  tab['mag_w1_vg']-tab['mag_w2_vg']]).T
    return X


# decontamination models

# no training  required but pass for consistency
def two_lines_eyeball(X_train, y_train, X_valid, fn_model=None):
    cut_0 = 0., 1., 0.2
    cut_1 = 1., 1., 2.9
    color_cuts = [cut_0, cut_1]

    # this predicts that all that make cuts are quasars, others are contaminants
    idx_predq = utils.gw1_w1w2_cuts_index(X_valid[:,0], X_valid[:,1], color_cuts)
    y_pred = np.full(X_valid.shape[0], 'o')
    y_pred[idx_predq] = 'q'

    return y_pred


def two_lines(X_train, y_train, X_valid, class_labels=None, fn_model=None):

    def objective(x):
        print(x)
        color_cuts = [[x[0], x[1], x[2]], [x[3], x[4], x[5]]]
        idx_predq = utils.gw1_w1w2_cuts_index(X_train[:,0], X_train[:,1], color_cuts)
        y_pred = np.full(X_train.shape[0], 'o')
        y_pred[idx_predq] = 'q'
        conf_mat = utils.confusion_matrix(y_pred, y_train, class_labels)
        #comp = utils.completeness(conf_mat, class_labels, label='q')
        fn = utils.N_FN(conf_mat, class_labels, label='q')
        fp = utils.N_FP(conf_mat, class_labels, label='q')
        #purity = utils.purity(conf_mat, class_labels, label='q')
        # dumb but trying
        return fp + fn

    #p0 = [0., 1., 0.2, 1., 1., 2.9]
    p0 = [0., 1., 0.5, 1., 1., 2.0]
    result = minimize(objective, p0, method='L-BFGS-B')
    x = result['x']
    color_cuts = [[x[0], x[1], x[2]], [x[3], x[4], x[5]]]
    idx_predq = utils.gw1_w1w2_cuts_index(X_valid[:,0], X_valid[:,1], color_cuts)
    y_pred = np.full(X_valid.shape[0], 'o')
    y_pred[idx_predq] = 'q'

    return y_pred


def svm(X_train, y_train, X_valid, fn_model=None):
    from sklearn import svm
    #clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=0.0001)
    # best: C=160-170
    clf = svm.LinearSVC(class_weight='balanced', C=160)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    print(y_pred.shape)
    print(y_pred)
    y_pred = np.array([yy.decode("utf-8") for yy in y_pred])
    if fn_model is not None:
        np.savez(fn_model, clf)
    return y_pred


if __name__=='__main__':
    main()