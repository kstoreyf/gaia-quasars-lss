import numpy as np

import healpy as hp
import george

import utils
import masks
import maps


def main():

    # TODO: remove constant in X; dont need for GP

    map_names = ['dust', 'stars', 'm10']
    NSIDE = 64
    #G_max = 19.9
    #G_max = 20.4
    G_max = 20.0
    fit_with_mask_mcs = False
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    fn_prob = f"../data/maps/map_probability_{'_'.join(map_names)}_NSIDE{NSIDE}_G{G_max}_testing.fits"
    overwrite = True

    print("Loading data")
    #fn_gaia = f'../data/gaia_G{G_max}.fits' 
    fn_gaia = f'../data/gaia_clean.fits' 
    tab_gaia = utils.load_table(fn_gaia)
    i_makeGcut = tab_gaia['phot_g_mean_mag'] < G_max
    tab_gaia = tab_gaia[i_makeGcut]

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
    # idx_fit = np.full(len(idx_fit), False)
    # idx_fit[:100] = True

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

    # TODO: save map!
    print("Making probability map")
    map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel)
    hp.write_map(fn_prob, map_prob, overwrite=overwrite)
    print(f"Saved map to {fn_prob}!")


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
                     'm10': maps.get_m10_map}

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


def construct_X(NPIX, map_names, maps_forsel):
    f_dict = {'dust': f_dust,
            'stars': f_stars,
            'm10': f_m10}

    constant = np.ones(NPIX)
    X_full = np.atleast_2d(constant)
    for map_name, map in zip(map_names, maps_forsel):
        X_full = np.vstack((X_full, f_dict[map_name](map)))
    return X_full.T


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
            X_scaled[:,1:] -= np.mean(X_scaled[:,1:], axis=0)
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
        # hyperparameter!
        kernel = george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
        self.gp = george.GP(kernel)
        self.gp.compute(self.X_train_scaled, self.y_err_train_scaled)

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred)
        y_pred_scaled, _ = self.gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)


if __name__=='__main__':
    main()
