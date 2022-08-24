import numpy as np
import os
import pandas as pd

import healpy as hp
from astropy import units as u
from astropy.table import Table


def jackknife(func, data, rand, *args, **kwargs):
    n = 12 # magic
    l_name = "l"
    assert np.all(data[l_name] >= 0.*u.deg) and np.all(data[l_name] < 360.*u.deg) # seriously; degrees?
    assert np.all(rand[l_name] >= 0.*u.deg) and np.all(rand[l_name] < 360.*u.deg)
    dl = 360. / n
    def one_jack(i):
        l1 = (i * dl)*u.deg
        l2 = ((i + 1) * dl)*u.deg
        idx_data = (data[l_name] < l1) | (data[l_name] >= l2)
        idx_rand = (rand[l_name] < l1) | (rand[l_name] >= l2)
        kwargs['jack'] = i
        return func(data[idx_data], rand[idx_rand], *args, **kwargs)
    outs = np.array(list(map(one_jack, range(n))))
    return jackknife_mean_var(outs)


def jackknife_mean_var(values):
    values = np.array(values)
    n = values.shape[0]
    print(f"Jackknife mean & var with n={n}")
    mean = np.mean(values, axis=0)
    var = ((n - 1) / n) * np.sum((values - mean) ** 2, axis=0)
    return mean, var


def get_fraction_recovered(Y_true, Y_hat, z_err_close):
        return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)


def add_gaia_wise_colors(tab):
    g = tab['phot_g_mean_mag']
    bp = tab['phot_bp_mean_mag']
    rp = tab['phot_rp_mean_mag']
    w1 = tab['w1mpro']
    w2 = tab['w2mpro']

    tab.add_column(g-rp, name='g_rp')
    tab.add_column(bp-g, name='bp_g')
    tab.add_column(bp-rp, name='bp_rp')
    tab.add_column(g-w1, name='g_w1')
    tab.add_column(w1-w2, name='w1_w2')


def add_spzs(tab_gaia, fn_spz='../data/redshifts_spz_kNN.fits'):
    tab_spz = Table.read(fn_spz, format='fits')
    assert np.allclose(tab_gaia['source_id'], tab_spz['source_id']), "Source IDs don't line up! They should by construction"
    tab_gaia.add_column(tab_spz['redshift_spz'], name='redshift_spz')
    tab_gaia.add_column(tab_spz['redshift_sdss'], name='redshift_sdss')
    

def load_table(fn_fits):
    return Table.read(fn_fits, format='fits')


def write_table(fn_table, data_cols, col_names, overwrite=False):
    tab = Table(data_cols, names=col_names)
    tab.write(fn_table, overwrite=overwrite)
    return tab


# copied from https://stackoverflow.com/questions/49372918/group-numpy-into-multiple-sub-arrays-using-an-array-of-values
def groupby(values, group_indices):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = group_indices.argsort(kind='mergesort')
    values_sorted = values[sidx]
    group_indices_sorted = group_indices[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,group_indices_sorted[1:] != group_indices_sorted[:-1],True])

    # Split input array with those start, stop ones
    values_grouped = [values_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return values_grouped, list(set(group_indices_sorted))




### Coordinates

# following https://mathworld.wolfram.com/SphericalCoordinates.html
def cartesian_to_spherical(x, y, z):
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arctan2(y,x)
    phi     =  np.arccos(z/r)
    return np.array([r, theta, phi])


def spherical_to_cartesian(r, theta, phi):
    x       =  r*np.cos(theta)*np.sin(phi)
    y       =  r*np.sin(theta)*np.sin(phi)
    z       =  r*np.cos(phi)
    return np.array([x, y, z])


def spherical_to_radec(theta, phi):
    ra = theta * 180/np.pi #+ 180
    #dec = phi * 180/np.pi - 90  
    print("utils.spherical_to_radec: changed this! fix surrounding code if this breaks")
    dec = 90 - phi * 180/np.pi
    return ra, dec 


# TODO: phi here isnt the inverse of above, check!
def radec_to_spherical(ra, dec):
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180
    return theta, phi


def cartesian_to_radec(x, y, z):
    _, theta, phi = cartesian_to_spherical(x, y, z)
    return spherical_to_radec(theta, phi)


def radec_to_cartesian(r, ra, dec):    
    theta, phi = radec_to_spherical(ra, dec)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    return np.array([x, y, z])


def random_ra_dec_on_sphere(rng, N_sphere):
    us = rng.random(size=N_sphere)
    vs = rng.random(size=N_sphere)
    theta_sphere = 2 * np.pi * us
    phi_sphere = np.arccos(2*vs-1)
    
    ra_sphere, dec_sphere = spherical_to_radec(theta_sphere, phi_sphere)
    return ra_sphere*u.deg, dec_sphere*u.deg


### Units

def Mpc_to_Mpcperh(distances_Mpc, cosmo):
    return distances_Mpc * cosmo.h


def Mpcperh_to_Mpc(distances_Mpcperh, cosmo):
    return distances_Mpcperh / cosmo.h


