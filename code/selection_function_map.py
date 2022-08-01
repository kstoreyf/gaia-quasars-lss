from astropy.table import Table, join
import numpy as np
import seaborn as sns
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import time
from numpy.random import default_rng

import healpy as hp
import pymaster as nmt
import pandas as pd

import utils
import generate_random
import correlations
import masks



def main():

    map_names = ['dust', 'stars', 'm10']


    NSIDE = 64
    G_max = 20
    fit_with_mask_mcs = True

    fn_gaia = f'../data/gaia_G{G_max}.fits' 
    tab_gaia = utils.load_table(fn_gaia)

    maps = load_maps(NSIDE, map_names)
    map_nqso_data, _ = utils.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)

    X_train_full = construct_X(NPIX, map_names, maps)
    y_train_full = map_nqso_data
    y_err_train_full = np.sqrt(y_train_full) # assume poission error

    NPIX = hp.nside2npix(NSIDE)

    x_scale_name = 'zeromean'
    y_scale_name = 'log'

    # should i do this in fitter??
    if y_scale_name=='log':
        idx_fit = y_full > 0 #hack
    else:
        idx_fit = np.full(len(y_full), True)

    if fit_with_mask_mcs:
        mask_mcs = masks.magellanic_clouds_mask(NSIDE)
        idx_fit = idx_fit & idx_nomcs

    X_train = X_train_full[idx_fit]
    y_train = X_train_full[idx_fit]
    y_err_train = y_err_train_full[idx_fit]
    
    fitter = FitterGP(X_train, y_train, y_train_err, 
                      x_scale_name=x_scale_name, y_scale_name=y_scale_name)
    fitter.train()
    y_pred = fitter.predict(X_train[idx_fit])

    y_pred_full = np.zeros(y_train_full.shape)
    y_pred_full[idx_fit] = y_pred

    # TODO: save map!

    def map_expected_to_probability(map_expected):
        # TODO: figure out how to get these maps/vals in here
        idx_clean = (map_dust < 0.03) & (map_stars < 15) & (map_m10 > 21)
        print("Number of clean pixels:", np.sum(idx_clean), f"(Total: {NPIX})")
        nqso_clean = np.mean(map_nqso_data[idx_clean])
        print(nqso_clean)

        map_prob = y_pred_gp_full_unscaled / nqso_clean
        map_prob[map_prob>1.0] = 1.0
        print(np.min(map_prob), np.max(map_prob))



def load_maps(NSIDE, map_names):
    maps = []
    for map_names in maps_names:
        fn_map = f'../data/maps/map_{map_name}_NSIDE{NSIDE}.npy'
        maps.append( np.load(fn_map) )
    return maps


def f_dust(map_d):
    return map_d


def f_stars(map_s):
    return np.log(map_s)


def f_m10(map_m):
    return map_m


def construct_X(NPIX, map_names, maps):
    f_dict = {'dust': f_dust,
            'stars': f_stars,
            'm10': f_m10}

    constant = np.ones(NPIX)
    X_full = np.atleast_2d(constant)
    for i, map_name in enumerate(map_names):
        X_full = np.vstack((X_full, f_dict[map_name](maps[i])))
    return X_full


class Fitter():

    def __init__(X_train, y_train, y_err_train, x_scale_name=None, y_scale_name=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_err_train = y_err_train
        self.x_scale_name = x_scale_name
        self.y_scale_name = y_scale_name

        self.X_scaled = self.scale_X(self.X_train)
        self.y_scaled = self.scale_y(self.y_train)
        self.y_err_train_scaled = self.scale_y_err(self.y_err_train)


    def scale_y_err(self, y_err)
        if self.y_scale_name=='log':
            return y_err / self.y
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
            y_scaled = np.log(y_scaled)
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
    def __init__(self):
        super().__init__()


    def train(self):
        ndim = self.X_train.shape[1]
        # hyperparameter!
        kernel = george.kernels.ExpSquaredKernel(1.0, ndim=ndim)
        gp = george.GP(kernel)
        gp.compute(self.X_train_scaled, self.y_train_err_scaled)

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred)
        y_pred_scaled = gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)


if __name__=='__main__':
    main()