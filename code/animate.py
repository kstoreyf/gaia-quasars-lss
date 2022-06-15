import numpy as np
from numpy.random import default_rng
import time

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from astropy.table import Table
import astropy.cosmology


def main():
    #fn_data = '../data/gaia_photoz.fits'
    fn_data = '/scratch/ksf293/gaia-quasars-lss/data/gaia_photoz.fits'
    format_save = 'gif'
    N_sub_str = '1e3'
    fn_save = f'../plots/animations/quasars_N{N_sub_str}.{format_save}'

    print("Reading data:", fn_data)
    data = Table.read(fn_data, format='fits')

    idx_pure = pure_cut(data)
    data = data[idx_pure]
    print("Loaded data with N =", len(data))

    # Subsample, if desired
    N_sub = int(N_sub_str)
    data = subsample(data, N_sub)
    print("Subsampled data to N =", len(data))

    # Convert to cartesian
    print("Converting to cartesian")
    add_xyz(data)

    # add color
    print("Adding color")
    add_g_rp_color(data)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)

    # Create an init function and the animate functions.
    def init():
        scat = ax.scatter(data['x'], data['y'], data['z'], s=1, alpha=0.5, c=data['g_rp'],
                        cmap='plasma', vmin=0, vmax=1)
        ax.axis('off')

        lim = 1500
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_facecolor('black')
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    print("Initializing animation")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)

    print("Saving animation to", fn_save)
     s = time.time()
    if format_save=='gif':
        anim.save(fn_save, writer=PillowWriter(fps=32))
    elif format_save=='mp4':
        anim.save(fn_save, fps=30, extra_args=['-vcodec', 'libx264'])
    e = time.time()
    print("Save time:", e-s)
    
    print("Saved!")


def add_xyz(data):
    cosmo = astropy.cosmology.Planck15
    dist_photoz = (cosmo.comoving_distance(data['redshift_photoz_est'])*cosmo.h).value # convert to Mpc/h
    data.add_column(dist_photoz, name='distance')

    x, y, z = radec_to_cartesian(data['distance'], data['ra'], data['dec'])
    data.add_column(x, name='x')
    data.add_column(y, name='y')
    data.add_column(z, name='z')


def add_g_rp_color(data):
    g_rp = np.array(data['phot_g_mean_mag']) - np.array(data['phot_rp_mean_mag'])
    data.add_column(g_rp, name='g_rp')


def subsample(data, N_sub):
    idx_sub = np.random.randint(low=0, high=len(data), size=N_sub)
    return data[idx_sub]


def pure_cut(table):
    idx_pure = np.where( (table['gaia_crf_source']==True) |
                         ((table['host_galaxy_flag']>0) & (table['host_galaxy_flag']<6)) |
                         (table['classlabel_dsc_joint']=='quasar') |
                         (table['vari_best_class_name']=='AGN') )[0]
    return idx_pure


def radec_to_cartesian(r, ra, dec):
    
    theta = ra * np.pi/180
    phi = (90 - dec) * np.pi/180

    x, y, z = spherical_to_cartesian(r, theta, phi)
    return np.array([x, y, z])


def spherical_to_cartesian(r, theta, phi):
    x       =  r*np.cos(theta)*np.sin(phi)
    y       =  r*np.sin(theta)*np.sin(phi)
    z       =  r*np.cos(phi)
    return np.array([x, y, z])


if __name__=='__main__':
    main()
