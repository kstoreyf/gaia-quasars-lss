import numpy as np
import os
from numpy.random import default_rng

from scipy.optimize import minimize, basinhopping

import utils


def main():

    # set params
    model = two_lines
    #model = svm
    lambda_rel = 0.1
    #fn_model = f'../data/decontamination_models/model_2lines_straight_lambda{lambda_rel}.npy'
    fn_model = f'../data/decontamination_models/conf_mats_2lines_straight_mag0.01.npy'
    exclude_gals = False

    # Full sample to apply to
    fn_gwnec = '../data/gaia_candidates_superset.fits'
    tab_gwnec = utils.load_table(fn_gwnec, format='fits')
    print(f"Number of Gaia quasar candidates with necessary data: {len(tab_gwnec)}")

    # Labeled sample
    fn_labeled = '../data/labeled_xsuperset.fits'
    tab_labeled = utils.load_table(fn_labeled)
    print(f"Number of labeled Gaia quasar candidates for training/validation: {len(tab_labeled)}")

    class_labels = ['q', 's', 'g']
    if exclude_gals:
        i_gals = tab_labeled['class'] == 'g'
        tab_labeled = tab_labeled[~i_gals]
        class_labels = ['q', 's']

    X_labeled = construct_X(tab_labeled)
    i_train, i_valid, i_test = utils.split_train_val_test(tab_labeled['rand_ints'], frac_train=0.5, frac_test=0.25, frac_val=0.25)
    
    X_train = X_labeled[i_train]
    X_valid = X_labeled[i_valid]

    y_labeled = tab_labeled['class']
    y_train = y_labeled[i_train]
    y_valid = y_labeled[i_valid]

    # Need class fractions bc otherwise treating stars and galaxies equally!
    class_nums = [np.sum(y_train==c) for c in class_labels]
    class_fracs = class_nums/np.sum(class_nums)
    print(class_fracs)

    y_valid_pred = model(X_train, y_train, X_valid, class_labels,
                        class_fracs, fn_model=fn_model, lambda_rel=lambda_rel)

    # print("Baseline: take all to be quasars")
    # y_valid_baseline = ['q']*len(y_valid)
    # compute_metrics(y_valid_baseline, y_valid, class_labels, class_fracs)

    print("Results")
    compute_metrics(y_valid_pred, y_valid, class_labels, class_fracs)


def compute_metrics(y_pred, y_true, class_labels, class_fracs):

    conf_mat = utils.confusion_matrix(y_pred, y_true, class_labels, class_fracs=class_fracs)
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


def two_lines(X_train, y_train, X_valid, class_labels, class_fracs,
              fn_model=None,
              lambda_rel=0.03, lambda_tp=10):

    class_nums = [np.sum(y_train==c) for c in class_labels]
    class_fracs = class_nums/np.sum(class_nums)

    def loss(y_pred, y_true):
        #conf_mat = utils.confusion_matrix(y_pred, y_true, class_labels, class_fracs=class_fracs)
        conf_mat = utils.confusion_matrix(y_pred, y_true, class_labels, class_fracs=class_fracs)
        #comp = utils.completeness(conf_mat, class_labels, label='q')
        #purity = utils.purity(conf_mat, class_labels, label='q')
        fn = utils.N_FN(conf_mat, class_labels, label='q')
        fp = utils.N_FP(conf_mat, class_labels, label='q')
        tp = utils.N_TP(conf_mat, class_labels, label='q')
        #print(comp, purity, tp)
        #print(fn, fp, tp)
        loss_val = lambda_rel*fp - tp
        print(loss_val, fp, tp)
        if np.abs(tp)<1e-6:
            print("bad!")
            loss_val = np.inf
        # fn + fp leads to it finding an area w no objects lol
        #return -(comp + lambda_rel*purity + lambda_tp*tp)
        #return fn + lambda_rel*fp - lambda_tp*tp
        return loss_val

    def get_cuts(g_w1, w1_w2, cuts):
        idx_predq = np.full(len(g_w1), True)
        for cut in cuts:
            # doing like this instead of y=mx+b so no numbers need to be huge to get big slope
            #idx_predq_single = w1_w2 > (cut[0] * g_w1 + cut[1])
            idx_predq_single = cut[0] * g_w1 + cut[1] * w1_w2 > cut[2]
            idx_predq = idx_predq & idx_predq_single
        return idx_predq

    def objective(x):
        #print(x)
        #color_cuts = [[x[0], x[1], x[2]], [x[3], x[4], x[5]]]
        color_cuts = [[0, 1, x[0]], [1, 0, x[1]]]

        #idx_predq = utils.gw1_w1w2_cuts_index(X_train[:,0], X_train[:,1], color_cuts)
        #color_cuts = [[x[0], x[1]], [x[2], x[3]]]
        idx_predq = utils.gw1_w1w2_cuts_index(X_train[:,0], X_train[:,1], color_cuts)
        y_pred = np.full(X_train.shape[0], 's') # label all star even tho some gals
        y_pred[idx_predq] = 'q'
        #return loss(y_pred, y_train)
        return utils.confusion_matrix(y_pred, y_train, class_labels)

    p0 = [0.0, 1.0]
    #result = minimize(objective, p0, method='Nelder-Mead')
    #result = basinhopping(objective, p0)
    #x = result['x']
    #print(result)
    #print(list(x))
    val_spacing = 0.01
    g_w1_vals = np.arange(1.5, 3+val_spacing, val_spacing)
    w1_w2_vals = np.arange(0, 1+val_spacing, val_spacing)
    #results = {}
    conf_mats = []
    for i in range(len(g_w1_vals)):
        conf_mats_inner = []
        for j in range(len(w1_w2_vals)):
            print(g_w1_vals[i], w1_w2_vals[j])
            # switch order here
            conf_mat = objective([w1_w2_vals[j], g_w1_vals[i]])
            #results[(g_w1_vals[i], w1_w2_vals[j])] = conf_mat
            conf_mats_inner.append(conf_mat)
        conf_mats.append(conf_mats_inner)

    res = {}
    res['grid'] = [g_w1_vals, w1_w2_vals]
    res['conf_mats'] = conf_mats
    np.save(fn_model, res)

    # color_cuts = [[0, 1, x[0]], [1, 0, x[1]]]
    # np.savetxt(fn_model, color_cuts)

    # idx_predq = get_cuts(X_valid[:,0], X_valid[:,1], color_cuts)
    # y_pred = np.full(X_valid.shape[0], 's')
    # y_pred[idx_predq] = 'q'

    # return y_pred


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