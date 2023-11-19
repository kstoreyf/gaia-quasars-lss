import argparse
import numpy as np
import os
import sys
import time

from scipy.optimize import minimize
import healpy as hp
import george

import utils
import masks
import maps

"""
Run the selection function with the following command on the commandline: 
python selection_function_map.py <fn_catalog> <fn_selfunc>

<fn_catalog>: the filepath to catalog to fit a selection function to. 
must be in fits format and have 'ra' and 'dec' columns in degrees

<fn_selfunc>: the filepath to the selection function map that will be saved;
should have .fits extension.

<fn_parentcat> (optional): the filepath to the "parent" catalog if relevant; used only to determine zero-pixels (pixels with no quasars) to mask out in final selection function, e.g. if computing the selection function for a redshift slice or other subsample that may have more zero-pixels than should be masked in the final map, only the zero-pixels in the parent catalog will end up being masked. should have .fits extension.

For more control, you can edit the parameters passed to the "run" function call
within parse_args.
These settings include: which feature maps to include, the NSIDE, which fitting model, the 
scalings of the features and label values, whether to overwrite maps (see run() function signature).
"""

def parse_args():
    parser=argparse.ArgumentParser(description="make selection function map for input catalog")
    parser.add_argument("fn_catalog", type=str)
    parser.add_argument("fn_selfunc", type=str)
    parser.add_argument("fn_parentcat", type=str, nargs='?', default=None)
    args=parser.parse_args()

    print(f"Running selection function with fn_catalog={args.fn_catalog}, fn_selfunc={args.fn_selfunc}, fn_parentcat={args.fn_parentcat}")
    run(args.fn_catalog, args.fn_selfunc, fn_parentcat=args.fn_parentcat)


def main():
    print("starting selection function", flush=True)

    tag_sel = ''
    G_max = 20.0
    tag_cat = ''
    fn_gaia = f'../data/quaia_G{G_max}{tag_cat}.fits' 
    fn_parentcat = None
    fitter_name = 'GP'

    map_names = ['dust', 'stars', 'm10', 'mcs', 'unwise', 'unwisescan', 'mcsunwise']
    NSIDE = 64
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    pixels_to_fit_mode = 'nonzero'
    y_err_mode = 'poisson'

    fit_mean = True
    overwrite_ypred = False
    overwrite_selfunc = True

    if not fit_mean:
        tag_sel += '_nofitmean'
    if fitter_name != 'GP':
        tag_sel += f'_{fitter_name}'
    fn_selfunc = f"../data/maps/selection_function_NSIDE{NSIDE}_G{G_max}{tag_cat}{tag_sel}.fits"

    run(fn_gaia, fn_selfunc, NSIDE=NSIDE, map_names=map_names, 
        fitter_name=fitter_name,
        x_scale_name=x_scale_name, y_scale_name=y_scale_name,
        y_err_mode=y_err_mode,
        pixels_to_fit_mode=pixels_to_fit_mode, fn_parentcat=fn_parentcat,
        fit_mean=fit_mean,
        overwrite_ypred=overwrite_ypred, overwrite_selfunc=overwrite_selfunc, 
        )


def run(fn_gaia, fn_selfunc, NSIDE=64, 
        map_names=['dust', 'stars', 'm10', 'mcs', 'unwise', 'unwisescan', 'mcsunwise'], 
        fitter_name='GP', x_scale_name='zeromean', y_scale_name='log',
        y_err_mode='poisson', 
        pixels_to_fit_mode='nonzero', fn_parentcat=None,
        fit_mean=True, overwrite_ypred=False, overwrite_selfunc=True):

    start = time.time()

    fn_ypred = f'{os.path.dirname(fn_selfunc)}/y_pred_{os.path.basename(fn_selfunc)}'
    if os.path.exists(fn_selfunc) and not overwrite_selfunc and os.path.exists(fn_ypred) and not overwrite_ypred:
        sys.exit(f"y_pred {fn_ypred} exists and overwrite_ypred is {overwrite_ypred}, and selection function {fn_selfunc} exists and overwrite is {overwrite_selfunc}, so exiting")

    print("Loading data", flush=True)
    tab_gaia = utils.load_table(fn_gaia)
    print("Making QSO map", flush=True)
    maps_forsel = load_maps(NSIDE, map_names)

    map_nqso_data, _ = maps.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)
    y_train_full = map_nqso_data
    # need this because will be inserting small vals where zero
    y_train_full = y_train_full.astype(float)
    i_nonzero = y_train_full > 0

    if os.path.exists(fn_ypred) and not overwrite_ypred:
        y_pred_full = hp.read_map(fn_ypred)
    else:

        print("Constructing X", flush=True)
        NPIX = hp.nside2npix(NSIDE)
        X_train_full = construct_X(NPIX, map_names, maps_forsel, fitter_name)

        print("Getting indices to fit", flush=True)
        if pixels_to_fit_mode=='all':
            if y_scale_name=='log':
                idx_zero = np.abs(y_train_full) < 1e-4
                print('num zeros:', np.sum(idx_zero))
                y_train_full[idx_zero] = 0.5       # set zeros to 1/2 a star
            idx_fit = np.full(len(y_train_full), True)
            print('min post', np.min(y_train_full), flush=True)
        elif pixels_to_fit_mode=='nonzero' :
            idx_fit = i_nonzero
        elif pixels_to_fit_mode=='okay':
            i_okay = get_okay_pixels(map_nqso_data, map_names, maps_forsel)
            idx_fit = i_nonzero & i_okay
        else:
            raise ValueError(f'mode {pixels_to_fit_mode} not recognized!')

        # poisson: standard dev = sqrt(N) [so var = N]
        if y_err_mode=='poisson':
            y_err_train_full = np.sqrt(y_train_full) # assume poission error
            print('err min max:', np.min(y_err_train_full), np.max(y_err_train_full))
        elif y_err_mode=='none':
            y_err_train_full = None
        else:
            raise ValueError(f'mode {y_err_mode} not recognized!')

        #FOR TESTING PURPOSES ONLY
        # print("TINY TEST")
        # idx_fit[100:] = False
        # print("THIRD TEST")
        # idx_fit[0::3] = False
        # idx_fit[1::3] = False

        X_train = X_train_full[idx_fit]
        y_train = y_train_full[idx_fit]
        if y_err_train_full is None:
            y_err_train = None
        else:
            y_err_train = y_err_train_full[idx_fit]

        print("Training fitter", flush=True)
        print("X_train:", X_train.shape, "y_train:", y_train.shape, flush=True)
        print("y_train min:", np.min(y_train), "y_train:", np.max(y_train), flush=True)
        fitter_dict = {'linear': FitterLinear,
                    'GP': FitterGP}
        fitter_class = fitter_dict[fitter_name]
        fitter = fitter_class(X_train, y_train, y_err_train, fitter_name,
                        fit_mean=fit_mean,
                        x_scale_name=x_scale_name, y_scale_name=y_scale_name,
                        map_names=map_names)
        fitter.train()
        
        print("Predicting", flush=True)
        y_pred_full = fitter.predict(X_train_full)
        
        print(f"Saving ypred map to {fn_ypred}")
        hp.write_map(fn_ypred, y_pred_full, overwrite=overwrite_ypred)

    print('RMSE:', utils.compute_rmse(y_pred_full, y_train_full), flush=True)

    if not os.path.exists(fn_selfunc) or overwrite_selfunc:
        print("Making selection function map", flush=True)
        map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel, NSIDE=NSIDE, i_keep=i_nonzero, fn_parentcat=fn_parentcat)

        hp.write_map(fn_selfunc, map_prob, overwrite=overwrite_selfunc)
        print(f"Saved map to {fn_selfunc}!", flush=True)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60.} min)", flush=True)


def get_okay_pixels(map_true, map_names, maps_forsel):
    okay_dict = {}    
    i_nonzero = map_true > 0
    for map_name, map_i in zip(map_names, maps_forsel):
        if map_name=='abssinb':
            continue
        if map_name=='m10':
            val = np.percentile(map_i[i_nonzero], 20)
        else:
            val = np.percentile(map_i[i_nonzero], 80)
        okay_dict[map_name] = val
    
    i_okay = np.full(len(map_true), True)
    for map_name, map_i in zip(map_names, maps_forsel):        
        if map_name=='m10':
            i_map_okay = map_i > okay_dict[map_name]
        elif map_name in ['dust', 'stars']:
            i_map_okay = map_i < okay_dict[map_name]
        else:
            continue
        i_okay = i_okay & i_map_okay
    print(f"{np.sum(i_okay)} of pixels have okay values of all templates!")
    print(f"That's {np.sum(i_okay)/len(i_okay)} of all pixels")
    print(f"And {np.sum(i_okay)/np.sum(i_nonzero)} of nonzero pixels")
    return i_okay


def map_expected_to_probability(map_expected, map_true, map_names, maps_forsel,
                                NSIDE=64, i_keep=None, fn_parentcat=None):
    idx_clean = np.full(len(map_expected), True)
    for map_name, map in zip(map_names, maps_forsel):
        if map_name=='dust':
            idx_map = map < 0.03
        elif map_name=='stars':
            idx_map = map < 15
        elif map_name=='m10':
            idx_map = map > 21
        elif map_name=='mcs':
            idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be (semi?-)redundant w stars
        elif map_name=='mcsunwise':
            idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be (semi?-)redundant w unwise
        idx_clean = idx_clean & idx_map
    print("Number of clean healpixels:", np.sum(idx_clean), f"(Total: {len(map_expected)})")
    nqso_clean = np.mean(map_true[idx_clean])
    # 2 standard deviations above should mean that most, if not all, values are below 1
    nqso_max = nqso_clean + 2*np.std(map_true[idx_clean])
    map_prob = map_expected / nqso_max

    if fn_parentcat is not None:
        print("Using parent catalog to determine zero-probability pixels")
        print("(If i_keep also passed, will be ignored!)")
        tab_parentcat = utils.load_table(fn_parentcat)
        map_nqso_parent, _ = maps.get_map(NSIDE, tab_parentcat['ra'], tab_parentcat['dec'], null_val=0)
        i_keep = map_nqso_parent>0

    if i_keep is not None:
        map_prob[~i_keep] = 0.0

    # it's alright if vals go above 1, it's just a rescaling, but with 2*std none of our fiducial catalogs have >1 vals
    print(f"min: {np.min(map_prob)}, max: {np.max(map_prob)}")
    return map_prob


def load_maps(NSIDE, map_names):
    maps_forsel = []
    # TODO: should be writing these maps with hp.write_map() to a fits file!
    map_functions = {'stars': maps.get_star_map,
                     'dust': maps.get_dust_map,
                     'm10': maps.get_m10_map,
                     'mcs': maps.get_mcs_map,
                     'unwise': maps.get_unwise_map,
                     'unwisescan': maps.get_unwise_scan_map,
                     'mcsunwise': maps.get_mcsunwise_map,
                     }

    for map_name in map_names:
        fn_map = f'../data/maps/map_{map_name}_NSIDE{NSIDE}.npy'
        maps_forsel.append( map_functions[map_name](NSIDE=NSIDE, fn_map=fn_map) )
    return maps_forsel


def f_dust(map_d):
    return map_d


def f_stars(map_s):
    return np.log(map_s)


def f_m10(map_m):
    return map_m


def f_mcs(map_mcs):
    map_mcs = map_mcs.astype(float)
    i_zeroorneg = map_mcs < 1e-4
    map_mcs[i_zeroorneg] = 1e-4
    return np.log(map_mcs)


def f_unwise(map_u):
    return np.log(map_u)


def f_unwisescan(map_us):
    return np.log(map_us)

def f_mcsunwise(map_mcsu):
    map_mcsu = map_mcsu.astype(float)
    i_zeroorneg = map_mcsu < 1e-4
    map_mcsu[i_zeroorneg] = 1e-4
    return np.log(map_mcsu)


def construct_X(NPIX, map_names, maps_forsel, fitter_name):
    f_dict = {'dust': f_dust,
             'stars': f_stars,
             'm10': f_m10,
             'mcs': f_mcs,
             'unwise': f_unwise,
             'unwisescan': f_unwisescan,
             'mcsunwise': f_mcsunwise,
             }
    X = np.vstack([f_dict[map_name](map) for map_name, map in zip(map_names, maps_forsel)])
    print(X.shape)
    if fitter_name=='linear':
        ones = np.full((1,X.shape[1]), 1)
        print(ones.shape)
        X = np.concatenate((ones, X))
    print(X.shape)
    return X.T


class Fitter():

    def __init__(self, X_train, y_train, y_err_train, fitter_name, 
                 x_scale_name=None, y_scale_name=None, map_names=None):
        # TODO: add asserts that these are the right shapes
        self.X_train = X_train
        self.y_train = y_train
        self.y_err_train = y_err_train
        # TODO: add asserts that these are implemented
        self.x_scale_name = x_scale_name
        self.y_scale_name = y_scale_name

        self.X_train_scaled = self.scale_X(self.X_train, fitter_name)
        self.y_train_scaled = self.scale_y(self.y_train)
        if y_err_train is None:
            self.y_err_train_scaled = None
        else:
            self.y_err_train_scaled = self.scale_y_err(self.y_err_train)
        self.fitter_name = fitter_name
        self.map_names = map_names


    def scale_y_err(self, y_err):
        if self.y_scale_name=='log':
            return y_err / self.y_train
        else:
            # if not log, make sure no zeros; set min to 1
            #hack!
            y_err = np.clip(y_err, 1, None)
        return y_err

    def scale_X(self, X, fitter_name):
        X_scaled = X.copy()
        if self.x_scale_name=='zeromean':
            if fitter_name=='linear':
                # careful, brittle! assumes first col is ones
                X_scaled[:,1:] -= np.mean(self.X_train[:,1:], axis=0)
            else:
                X_scaled -= np.mean(self.X_train, axis=0)
        return X_scaled


    def scale_y(self, y):
        y_scaled = y.copy()
        if self.y_scale_name=='log':
            y_scaled = np.log(y)
        return y_scaled


    def unscale_y(self, y_scaled):
        y_unscaled = y_scaled.copy()
        if self.y_scale_name=='log':
            y_unscaled = np.exp(y_scaled)
        return y_unscaled


    def train(self):
        pass


    def predict(self, X_pred):
        pass

# copied from abby williams
# https://github.com/abbyw24/lss-dipoles/blob/master/sel_func_map_with_dip.py
# class DipoleModel(Model):
#         """
#         BUGS:
#         - This class will only operate if there's a global variable called NSIDE
#         and a global variable called PIXEL_INDICES_TO_FIT.
#         """
#         parameter_names = ['monopole', 'dipole_x', 'dipole_y', 'dipole_z']
#         thetas, phis = hp.pix2ang(NSIDE, ipix=PIXEL_INDICES_TO_FIT)
        
#         def get_value(self, X):                        
#             return self.monopole + dipole(DipoleModel.thetas, DipoleModel.phis,
#                                         self.dipole_x, self.dipole_y, self.dipole_z) # this value has shape (len(PIXEL_INDICES_TO_FIT),)
        
#         def set_vector(self, v):
#             self.monopole, self.dipole_x, self.dipole_y, self.dipole_z = v


class FitterGP(Fitter):
    def __init__(self, *args, fit_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        print("fit_mean =", fit_mean)
        self.fit_mean = fit_mean


    def train(self):
        ndim = self.X_train.shape[1]
        n_params = self.X_train_scaled.shape[1]
        print("n params:", n_params)
        #based on previous optimizations (G20.0 most recently)
        log_init_guesses = {'dust': -0.5,
                        'stars': 1.5,
                        'm10': -1,
                        'mcs': 5,
                        'unwise': 0.8,
                        'unwisescan': 0,
                        'mcsunwise': 10,
                        }
        if self.map_names is not None:
            log_p0 = np.array([log_init_guesses[map_name] for map_name in self.map_names]) 
        else:
            log_p0 = np.full(0.1, n_params)
        p0 = np.exp(log_p0)
        kernel = george.kernels.ExpSquaredKernel(p0, ndim=ndim)

        # this will add a free parameter to scale the overall amplitude, which has units of variance
        amp = np.var(self.y_train_scaled)
        kernel *= amp
        
        # Using an initial guess of mean=0 terminates successfully more often,
        # than using the mean of the labels
        #mean = np.mean(self.y_train_scaled)
        mean = 0.0
        self.gp = george.GP(kernel, mean=mean, fit_mean=self.fit_mean)

        print('parameter names:', self.gp.get_parameter_names())
        print('paramters init:', self.gp.get_parameter_vector())

        print('y minmax', np.min(self.y_train_scaled), np.max(self.y_train_scaled))
        print('yerr:', self.y_err_train, self.y_err_train_scaled)

        if self.y_err_train_scaled is None:
            self.gp.compute(self.X_train_scaled)
        else:
            self.gp.compute(self.X_train_scaled, self.y_err_train_scaled)
        print('p compute:', self.gp.get_parameter_vector())
        print('lnlike compute:', self.gp.log_likelihood(self.y_train_scaled))

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            #print(p)
            #print('like:', -self.gp.log_likelihood(self.y_train_scaled))
            return -self.gp.log_likelihood(self.y_train_scaled)

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            #print(p)
            #print('grad like:', -self.gp.grad_log_likelihood(self.y_train_scaled))
            return -self.gp.grad_log_likelihood(self.y_train_scaled)

        print("Minimizing")
        result = minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like, 
                          #method='Powell'
                          )
        print(result)
        self.gp.set_parameter_vector(result.x)
        print('p post op:', self.gp.get_parameter_vector())
        print('lnlike final:', self.gp.log_likelihood(self.y_train_scaled))

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred, self.fitter_name)
        print('predict p:', self.gp.get_parameter_vector())
        y_pred_scaled, _ = self.gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)




class FitterLinear(Fitter):
    def __init__(self, *args, fit_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        print("fit_mean =", fit_mean)
        self.fit_mean = fit_mean


    def train(self):

        # ndim = self.X_train.shape[1]
        # n_params = self.X_train_scaled.shape[1]

        #theta = (X^T Cinv X) (X^T Cinv y)
        # For now, assume C is the idendity and omit
        if self.y_err_train_scaled is None:
            XTCinvX = self.X_train_scaled.T @ self.X_train_scaled
            XTCinvy = self.X_train_scaled.T @ self.y_train_scaled    
        else:        
            Cinv = np.diag(1./(self.y_err_train_scaled**2))
            XTCinvX = self.X_train_scaled.T @ Cinv @ self.X_train_scaled
            XTCinvy = self.X_train_scaled.T @ Cinv @ self.y_train_scaled

        res = np.linalg.lstsq(XTCinvX, XTCinvy, rcond=None)
        self.theta = res[0]
        print('theta:', self.theta)

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred, self.fitter_name)
        y_pred_scaled = X_pred_scaled @ self.theta
        return self.unscale_y(y_pred_scaled)


if __name__=='__main__':
    #main()
    parse_args()


