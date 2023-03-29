import itertools
import numpy as np
import os
import time
from numpy.random import default_rng

from scipy.optimize import minimize, basinhopping

import utils


def main():
    tag_decontam = '_mag0.01_lg1'
    #tag_decontam = '_mag0.1-0.01_lg1'
    overwrite_conf_mats = True
    fn_conf_mats = f'../data/decontamination_models/conf_mats{tag_decontam}.npy'
    fn_cuts = f'../data/color_cuts{tag_decontam}.txt'

    s = time.time()
    compute(fn_conf_mats, fn_cuts, overwrite_conf_mats=overwrite_conf_mats)
    overwrite_table = True
    #apply_to_quasar_catalog(fn_cuts, overwrite=overwrite_table)
    #apply_to_labeled_table(fn_cuts, overwrite=overwrite_table)
    e = time.time()
    print(f"Time: {e-s} s ({(e-s)/60} min)")


def compute(fn_conf_mats, fn_cuts, overwrite_conf_mats=False):

    # Full sample to apply to
    fn_gsup = '../data/gaia_candidates_superset.fits'
    tab_gsup = utils.load_table(fn_gsup, format='fits')
    print(f"Number of Gaia quasar candidates with necessary data: {len(tab_gsup)}")

    # Labeled sample
    fn_labeled = '../data/labeled_superset.fits'
    tab_labeled = utils.load_table(fn_labeled)
    print(f"Number of labeled Gaia quasar candidates for training/validation: {len(tab_labeled)}")
    class_labels = ['q', 's', 'g']
    print(tab_labeled.columns)

    # Make labeled training and validation data
    color_names = ['g_w1', 'w1_w2', 'bp_g']
    X_labeled = construct_X(tab_labeled, color_names)
    i_train, i_valid, i_test = utils.split_train_val_test(tab_labeled['rand_ints'], frac_train=0.5, frac_test=0.25, frac_val=0.25)
    
    X_train = X_labeled[i_train]
    X_valid = X_labeled[i_valid]

    y_labeled = tab_labeled['class']
    y_train = y_labeled[i_train]
    y_valid = y_labeled[i_valid]

    # Get cuts
    if not os.path.exists(fn_conf_mats) or overwrite_conf_mats:
        make_cut_grid(X_train, y_train, class_labels, color_names,
                     fn_conf_mats=fn_conf_mats)

    get_best_cuts(fn_conf_mats, class_labels, fn_cuts=fn_cuts)
    print("Done!")

    
def apply_to_quasar_catalog(fn_cuts, overwrite=False):
    fn_orig = '../data/gaia_candidates_superset.fits'
    fn_clean = '../data/gaia_candidates_clean.fits'
    make_clean_subsample(fn_cuts, fn_orig, fn_clean, overwrite=overwrite)
    

def apply_to_labeled_table(fn_cuts, overwrite=False):
    fn_orig = '../data/labeled_superset.fits'
    fn_clean = '../data/labeled_clean.fits'
    make_clean_subsample(fn_cuts, fn_orig, fn_clean, overwrite=overwrite)
    


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


def construct_X(tab, color_names):
    color_dict = {'g_w1': ['phot_g_mean_mag', 'mag_w1_vg'],
                  'w1_w2': ['mag_w1_vg', 'mag_w2_vg'],
                  'bp_g': ['phot_bp_mean_mag', 'phot_g_mean_mag']
                 }
    X = []
    for color_name in color_names:
        color_1, color_2 = color_dict[color_name]
        X.append(tab[color_1]-tab[color_2])
    X = np.array(X).T
    return X


def make_cut_grid(X_train, y_train, class_labels, color_names,
                  fn_conf_mats=None):

    def get_conf_mat(cuts_min):
        # transpose matrix because first arg is colors; cuts_min should be in proper order
        idx_predq = utils.cuts_index(X_train.T, cuts_min)
        y_pred = np.full(X_train.shape[0], 's') # label all star even tho some gals
        y_pred[idx_predq] = 'q'
        return utils.confusion_matrix(y_pred, y_train, class_labels)

    N_colors = len(color_names)
    cut_spacings = {'g_w1': 0.01,
                    'w1_w2': 0.01,
                    'bp_g': 0.01
                 }
    limits = {'g_w1': (1.75, 2.75),
              'w1_w2': (0.0, 1.0),
              'bp_g': (-1.0, 0.0)
             }
    cut_arr = [np.arange(limits[cn][0], limits[cn][1]+cut_spacings[cn], cut_spacings[cn]) for cn in color_names]
    index_ranges = [np.arange(len(cuts)) for cuts in cut_arr]

    # maybe a better way to do this but
    conf_mats = np.empty((*[len(cuts) for cuts in cut_arr], len(class_labels), len(class_labels)))
    for indices in itertools.product(*index_ranges):
        print(indices)
        cuts_min = [cut_arr[cc][indices[cc]] for cc in range(N_colors)]
        print(cuts_min)
        conf_mats[indices] = get_conf_mat(cuts_min)

    # val_spacing = 0.1
    # val_spacing_fine = 0.01
    # g_w1_vals = np.arange(1.75, 2.75+val_spacing, val_spacing)
    # w1_w2_vals = np.arange(0.0, 1.0+val_spacing_fine, val_spacing_fine)
    # bp_g_vals = np.arange(-1, 0.0+val_spacing, val_spacing)
    # conf_mats = np.empty((len(g_w1_vals), len(w1_w2_vals), len(bp_g_vals),
    #                      len(class_labels), len(class_labels)))
    # for i in range(len(g_w1_vals)):
    #     for j in range(len(w1_w2_vals)):
    #         for k in range(len(bp_g_vals)):
    #             print(g_w1_vals[i], w1_w2_vals[j], bp_g_vals[k])
    #             conf_mats[i,j,k] = get_conf_mat([g_w1_vals[i], w1_w2_vals[j], bp_g_vals[k]])

    results = {}
    results['cut_min_arr'] = cut_arr
    results['color_names'] = color_names
    results['conf_mats'] = np.array(conf_mats)
    np.save(fn_conf_mats, results)



def get_metric_matrices(fn_conf_mats, class_labels):
    results = np.load(fn_conf_mats, allow_pickle=True).item()
    cut_arr = results['cut_min_arr']
    color_names = results['color_names']
    conf_mats = np.array(results['conf_mats'])
    N_cuts = [len(cuts) for cuts in cut_arr]
    print(N_cuts)
    index_ranges = [np.arange(N_cut) for N_cut in N_cuts]

    tps = np.empty(N_cuts)
    fps_s = np.empty(N_cuts)
    fps_g = np.empty(N_cuts)

    i_q = class_labels.index('q')
    i_s = class_labels.index('s')
    i_g = class_labels.index('g')

    for indices in itertools.product(*index_ranges):
        conf_mat = conf_mats[indices]
        tps[indices] = utils.N_TP(conf_mat, class_labels, label='q')        
        fps_s[indices] = conf_mat[i_s,i_q]        
        fps_g[indices] = conf_mat[i_g,i_q]

    return tps, fps_s, fps_g


def objective_function(tps, fps_s, fps_g, lambda_s=3, lambda_g=1):
    return tps - lambda_s*fps_s - lambda_g*fps_g


def get_best_cuts(fn_conf_mats, class_labels, fn_cuts=None):

    results = np.load(fn_conf_mats, allow_pickle=True).item()
    color_names = results['color_names']
    cut_arr = results['cut_min_arr']

    tps, fps_s, fps_g = get_metric_matrices(fn_conf_mats, class_labels)
    objs = objective_function(tps, fps_s, fps_g)
    indices_best = np.unravel_index(objs.argmax(), objs.shape)
    N_colors = len(color_names)    
    cuts_min_best = [cut_arr[cc][indices_best[cc]] for cc in range(N_colors)]
    print(objs[indices_best])
    print(cuts_min_best)

    res = np.array([color_names, cuts_min_best]).T
    if fn_cuts is not None:
        np.savetxt(fn_cuts, res, fmt="%s")
    return cuts_min_best, indices_best


def make_clean_subsample(fn_cuts, fn_orig, fn_clean, overwrite=False):
    
    # Load data
    tab_orig = utils.load_table(fn_orig)
    color_names, cuts = np.genfromtxt(fn_cuts, dtype=['U15', '<f8'], unpack=True)
    print(color_names, cuts)

    # this will make sure color_names and cuts remain in proper order
    X_orig = construct_X(tab_orig, color_names)
    i_makes_cuts = utils.cuts_index(X_orig.T, cuts)
    tab_clean = tab_orig[i_makes_cuts]
    
    # Add random vals
    rng = np.random.default_rng(seed=42)
    tab_clean['rand_ints'] = rng.choice(range(len(tab_clean)), size=len(tab_clean), replace=False)

    # save
    print("N_clean:", len(tab_clean))
    tab_clean.write(fn_clean, overwrite=overwrite)


if __name__=='__main__':
    main()
