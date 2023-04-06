import numpy as np
import time

from scipy.optimize import minimize
import healpy as hp
import george

import utils
import masks
import maps


def main():


    map_names = ['dust', 'stars', 'm10', 'mcs']
    #map_names = ['dust', 'stars']
    NSIDE = 64
    G_max = 20.5
    #G_max = 20.0
    fit_with_mask_mcs = False
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    #fn_prob = f"../data/maps/mapQ_probability_{'_'.join(map_names)}_NSIDE{NSIDE}_G{G_max}_testing.fits"
    #fn_prob = f"../data/maps/mapQ_probability_{'_'.join(map_names)}_NSIDE{NSIDE}_G{G_max}_op_p0nparams.fits"
    #fn_prob = f"../data/maps/map_probability_{'_'.join(map_names)}_NSIDE{NSIDE}_G{G_max}_runlong.fits"
    fn_prob = f"../data/maps/selection_function_NSIDE{NSIDE}_G{G_max}.fits"
    overwrite = True

    start = time.time()

    print("Loading data")
    fn_gaia = f'../data/catalog_G{G_max}.fits' 
    #fn_gaia = f'../data/gaia_clean.fits' 
    tab_gaia = utils.load_table(fn_gaia)

    print("Making QSO map")
    maps_forsel = load_maps(NSIDE, map_names)
    map_nqso_data, _ = maps.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)

    print("Constructing X and y")
    NPIX = hp.nside2npix(NSIDE)
    X_train_full = construct_X(NPIX, map_names, maps_forsel)
    y_train_full = map_nqso_data
    y_err_train_full = np.sqrt(y_train_full) # assume poission error

    print("Getting indices to fit")
    # should i do this in fitter??
    if y_scale_name=='log':
        idx_fit = y_train_full > 0 #hack
    else:
        idx_fit = np.full(len(y_train_full), True)

    if fit_with_mask_mcs:
        mask_mcs = masks.magellanic_clouds_mask(NSIDE)
        idx_nomcs = ~mask_mcs #because mask is 1 where masked
        # i think nomcs is breaking everything! #TODO check
        idx_fit = idx_fit & idx_nomcs

    # FOR TEST
    #idx_fit[10000:] = False

    print()

    X_train = X_train_full[idx_fit]
    y_train = y_train_full[idx_fit]
    y_err_train = y_err_train_full[idx_fit]

    print("Training fitter")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    fitter = FitterGP(X_train, y_train, y_err_train, 
                      x_scale_name=x_scale_name, y_scale_name=y_scale_name)
    fitter.train()
    print("Predicting")
    y_pred = fitter.predict(X_train)

    y_pred_full = np.zeros(y_train_full.shape)
    y_pred_full[idx_fit] = y_pred

    print('RMSE:', utils.compute_rmse(y_pred_full, y_train_full))

    print("Making probability map")
    map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel)
    hp.write_map(fn_prob, map_prob, overwrite=overwrite)
    print(f"Saved map to {fn_prob}!")

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60.} min)")


#hack! better way?
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
                     'mcs': maps.get_mcs_map}

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
    #rng = np.random.default_rng(42)
    #map_mcs += rng.normal(loc=0, scale=0.1, size=len(map_mcs))
    print(np.min(map_mcs), np.max(map_mcs))
    #i_zeroorneg = map_mcs < 1.001
    #map_mcs[i_zeroorneg] = 1.001
    i_zeroorneg = map_mcs < 1e-4
    map_mcs[i_zeroorneg] = 1e-4
    print(np.min(map_mcs), np.max(map_mcs))
    print(np.min(np.log(map_mcs)), np.max(np.log(map_mcs)))
    return np.log(map_mcs)


def construct_X(NPIX, map_names, maps_forsel):
    f_dict = {'dust': f_dust,
             'stars': f_stars,
             'm10': f_m10,
             'mcs': f_mcs}

    # constant = np.ones(NPIX)
    # X_full = np.atleast_2d(constant)
    # for map_name, map in zip(map_names, maps_forsel):
    #     X_full = np.vstack((X_full, f_dict[map_name](map)))
    # return X_full.T
    X = np.vstack([f_dict[map_name](map) for map_name, map in zip(map_names, maps_forsel)])
    return X.T


class Fitter():

    def __init__(self, X_train, y_train, y_err_train, x_scale_name=None, y_scale_name=None):
        # TODO: add asserts that these are the right shapes
        self.X_train = X_train
        self.y_train = y_train
        self.y_err_train = y_err_train
        # TODO: add asserts that these are implemented
        self.x_scale_name = x_scale_name
        self.y_scale_name = y_scale_name

        self.X_train_scaled = self.scale_X(self.X_train)
        self.y_train_scaled = self.scale_y(self.y_train)
        self.y_err_train_scaled = self.scale_y_err(self.y_err_train)


    def scale_y_err(self, y_err):
        if self.y_scale_name=='log':
            return y_err / self.y_train
        else:
            # if not log, make sure no zeros; set min to 1
            #hack!
            y_err = np.clip(y_err, 1, None)
        return y_err

    def scale_X(self, X):
        X_scaled = X.copy()
        if self.x_scale_name=='zeromean':
            # assumes starts with constant feature! #hack
            #X_scaled[:,1:] -= np.mean(X_scaled[:,1:], axis=0)
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


class FitterGP(Fitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self):
        ndim = self.X_train.shape[1]
        n_params = self.X_train_scaled.shape[1]
        print("n params:", n_params)
        p0 = np.exp(np.full(n_params, 0.1))
        #p0 = 0.1
        kernel = george.kernels.ExpSquaredKernel(p0, ndim=ndim)

        self.gp = george.GP(kernel)
        print('p init:', self.gp.get_parameter_vector())
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
        result = minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like)
        print(result)
        self.gp.set_parameter_vector(result.x)
        print('p post op:', self.gp.get_parameter_vector())
        print('lnlike final:', self.gp.log_likelihood(self.y_train_scaled))

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred)
        print('predict p:', self.gp.get_parameter_vector())
        y_pred_scaled, _ = self.gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)


if __name__=='__main__':
    main()


