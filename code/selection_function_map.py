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

<fn_selfunc> (optional): the filepath to the selection function map that will be saved;
should have .fits extension. if not given, 'selection_function.fits' will be used

For more control, you can edit the settings in the "run" function call
within parse_args.
These settings are: which feature maps to include, the NSIDE, and the 
scalings of the features and label values (see run() function signature).
"""

def parse_args():
    parser=argparse.ArgumentParser(description="make selection function map for input catalog")
    parser.add_argument("fn_catalog", type=str, nargs='?')
    parser.add_argument("fn_selfunc", type=str, nargs='?', default='selection_function.fits')
    args=parser.parse_args()

    if args.fn_catalog is None:
        main()    
    else:
        print(f"Running selection function with fn_catalog={args.fn_catalog}, fn_selfunc={args.fn_selfunc}")
        run(args.fn_catalog, args.fn_selfunc)


def main():
    print("starting selection function", flush=True)

    tag_sel = '_unwise'
    G_max = 20.0
    tag_cat = ''
    #tag_cat = '_zsplit3bin2'f
    fn_gaia = f'../data/quaia_G{G_max}{tag_cat}.fits' 
    fitter_name = 'GP'

    #map_names = ['dust', 'stars', 'm10']
    map_names = ['dust', 'stars', 'm10', 'mcs', 'unwise']
    NSIDE = 64
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    fit_zeros = False
    #fit_mean = False
    fit_mean = True
    downweight_mcs = False
    overwrite = True

    if not fit_mean:
        tag_sel += '_nofitmean'
    if downweight_mcs:
        tag_sel += '_dwmcs'
    if fitter_name != 'GP':
        tag_sel += f'_{fitter_name}'
    fn_selfunc = f"../data/maps/selection_function_NSIDE{NSIDE}_G{G_max}{tag_cat}{tag_sel}.fits"

    fn_ypred = f"../data/maps/y_pred_NSIDE{NSIDE}_G{G_max}{tag_cat}{tag_sel}.fits"

    run(fn_gaia, fn_selfunc, NSIDE=NSIDE, map_names=map_names, 
        fitter_name=fitter_name,
        x_scale_name=x_scale_name, y_scale_name=y_scale_name,
        fit_zeros=fit_zeros, fit_mean=fit_mean, downweight_mcs=downweight_mcs,
        fn_ypred=fn_ypred, overwrite=overwrite, 
        )


def run(fn_gaia, fn_selfunc, NSIDE=64, map_names=['dust', 'stars', 'm10', 'mcs'], 
        fitter_name='GP', x_scale_name='zeromean', y_scale_name='log',
        fit_zeros=False, fit_mean=True, downweight_mcs=False,
        fn_ypred=None, overwrite=True):

    if os.path.exists(fn_selfunc) and not overwrite:
        sys.exit(f"Selection function path {fn_selfunc} exists and overwrite is {overwrite}, so exiting")

    start = time.time()

    print("Loading data", flush=True)
    tab_gaia = utils.load_table(fn_gaia)

    print("Making QSO map", flush=True)
    maps_forsel = load_maps(NSIDE, map_names)
    map_nqso_data, _ = maps.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)

    print("Constructing X and y", flush=True)
    NPIX = hp.nside2npix(NSIDE)
    X_train_full = construct_X(NPIX, map_names, maps_forsel, fitter_name)
    y_train_full = map_nqso_data
    # need this because will be inserting small vals where zero
    y_train_full = y_train_full.astype(float)

    print("Getting indices to fit", flush=True)
    if fit_zeros:
        if y_scale_name=='log':
            idx_zero = np.abs(y_train_full) < 1e-4
            print('num zeros:', np.sum(idx_zero))
            y_train_full[idx_zero] = 0.5       # set zeros to 1/2 a star
        idx_fit = np.full(len(y_train_full), True)
        print('min post', np.min(y_train_full), flush=True)
    else:
        idx_fit = y_train_full > 0

    # poisson: standard dev = sqrt(N) [so var = N]
    y_err_train_full = np.sqrt(y_train_full) # assume poission error
    print(np.min(y_err_train_full), np.max(y_err_train_full))
    if downweight_mcs:
        assert 'mcs' in map_names, "Need MCs map to downweight them [for now!]"
        idx_mcsmap = map_names.index('mcs')
        map_mcs = maps_forsel[idx_mcsmap]
        i_inmcs = map_mcs > 0
        factor_downweight_mcs = 1000 #magic number! ??
        y_err_train_full[i_inmcs] *= factor_downweight_mcs
        print(np.min(y_err_train_full), np.max(y_err_train_full))

    #FOR TESTING PURPOSES ONLY
    #print("TINY TEST")
    #idx_fit[100:] = False
    # print("THIRD TEST")
    # idx_fit[0::3] = False
    # idx_fit[1::3] = False

    X_train = X_train_full[idx_fit]
    y_train = y_train_full[idx_fit]
    y_err_train = y_err_train_full[idx_fit]

    print("Training fitter", flush=True)
    print("X_train:", X_train.shape, "y_train:", y_train.shape, flush=True)
    print(X_train[3:])
    fitter_dict = {'linear': FitterLinear,
                   'GP': FitterGP}
    fitter_class = fitter_dict[fitter_name]
    fitter = fitter_class(X_train, y_train, y_err_train, fitter_name,
                      fit_mean=fit_mean,
                      x_scale_name=x_scale_name, y_scale_name=y_scale_name,
                      map_names=map_names)
    fitter.train()
    print("Predicting", flush=True)
    y_pred = fitter.predict(X_train)

    y_pred_full = np.zeros(y_train_full.shape)
    y_pred_full[idx_fit] = y_pred
    
    if fn_ypred is not None:
        print(f"Saving ypred map to {fn_ypred}")
        hp.write_map(fn_ypred, y_pred_full, overwrite=overwrite)

    print('RMSE:', utils.compute_rmse(y_pred_full, y_train_full), flush=True)

    print("Making probability map", flush=True)
    map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel)
    hp.write_map(fn_selfunc, map_prob, overwrite=overwrite)
    print(f"Saved map to {fn_selfunc}!", flush=True)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60.} min)", flush=True)


def map_expected_to_probability(map_expected, map_true, map_names, maps_forsel):
    idx_clean = np.full(len(map_expected), True)
    for map_name, map in zip(map_names, maps_forsel):
        if map_name=='dust':
            idx_map = map < 0.03
        elif map_name=='stars':
            idx_map = map < 15
        elif map_name=='m10':
            idx_map = map > 21
        elif map_name=='mcs':
            idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be redundant w stars
        idx_clean = idx_clean & idx_map
    print("Number of clean healpixels:", np.sum(idx_clean), f"(Total: {len(map_expected)})")
    nqso_clean = np.mean(map_true[idx_clean])

    map_prob = map_expected / nqso_clean
    map_prob[map_prob>1.0] = 1.0
    assert np.all(map_prob <= 1.0) and np.all(map_prob >= 0.0), "Probabilities must be <=1 and >=0!"
    return map_prob


def load_maps(NSIDE, map_names):
    maps_forsel = []
    # TODO: should be writing these maps with hp.write_map() to a fits file!
    map_functions = {'stars': maps.get_star_map,
                     'dust': maps.get_dust_map,
                     'm10': maps.get_m10_map,
                     'mcs': maps.get_mcs_map,
                     'unwise': maps.get_unwise_map,
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


def construct_X(NPIX, map_names, maps_forsel, fitter_name):
    f_dict = {'dust': f_dust,
             'stars': f_stars,
             'm10': f_m10,
             'mcs': f_mcs,
             'unwise': f_unwise,
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
                X_scaled[:,1:] -= np.mean(X_scaled[:,1:], axis=0)
            else:
                X_scaled -= np.mean(X_scaled, axis=0)
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
        #based on previous optimizations
        log_init_guesses = {'dust': -0.1,
                        'stars': 2,
                        'm10': -2,
                        'mcs': 5,
                        'unwise': 1,
                        }
        if self.map_names is not None:
            log_p0 = np.array([log_init_guesses[map_name] for map_name in self.map_names]) 
        else:
            log_p0 = np.full(0.1, n_params)
        p0 = np.exp(log_p0)
        kernel = george.kernels.ExpSquaredKernel(p0, ndim=ndim)

        #print("using hodlr solver")
        #self.gp = george.GP(kernel, solver=george.HODLRSolver)
        # Initial mean 0 terminates successfully more often 
        #mean = np.mean(self.y_train_scaled)
        mean = 0.0
        self.gp = george.GP(kernel, mean=mean, fit_mean=self.fit_mean)
        print('p init:', self.gp.get_parameter_vector())
        # print(self.X_train_scaled)
        # print(self.y_err_train_scaled)

        self.gp.compute(self.X_train_scaled, self.y_err_train_scaled)
        print('p compute:', self.gp.get_parameter_vector())
        print('lnlike compute:', self.gp.log_likelihood(self.y_train_scaled))

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.log_likelihood(self.y_train_scaled)

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
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
        Cinv = np.diag(1./(self.y_err_train_scaled**2))
        XTCinvX = self.X_train_scaled.T @ Cinv @ self.X_train_scaled
        XTCinvy = self.X_train_scaled.T @ Cinv @ self.y_train_scaled
        # XTCinvX = self.X_train_scaled.T @ self.X_train_scaled
        # XTCinvy = self.X_train_scaled.T @ self.y_train_scaled
        #self.theta = np.linalg.solve(XTCinvX, XTCinvy)
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


