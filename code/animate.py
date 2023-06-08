import numpy as np
from numpy.random import default_rng
import time

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker

from astropy.table import Table
import astropy.cosmology
from astropy import units as u

import utils


def main():
    #fn_data = '../data/gaia_photoz.fits'
    #fn_data = '/scratch/ksf293/gaia-quasars-lss/data/gaia_photoz.fits'
    #fn_data = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'
    G_max = 20.5
    #data_tag = f'gcathi'
    data_tag = 'sdss'

    plot_dir = '../plots/2023-04-04_figures'
    anim_dir = '../plots/2023-05-27_animations'

    if 'gcat' in data_tag:
        fn_data = f'../data/QUaia_G{G_max}.fits'
        #title = f'QUaia ($G<{G_max}$)'
        title = None
        colorbar_label = r'redshift $z$'
        redshift_name = 'redshift_spz'
        property_colorby = redshift_name
        ra_name, dec_name = 'ra', 'dec'
    elif 'sdss' in data_tag:
        fn_data = f'../data/quasars_sdss_xgaiaall_xunwiseall_good.fits'
        title = None
        #title = 'SDSS DR16Q'
        colorbar_label = r'redshift $z$'
        redshift_name = 'z_sdss'
        property_colorby = redshift_name
        ra_name, dec_name = 'ra_sdss', 'dec_sdss'

    #format_save = 'gif'
    format_save = 'mp4'
    #N_sub_str = '1e3'
    N_sub_str = 'all'
    
    fn_save = f'{anim_dir}/{data_tag}_N{N_sub_str}_{property_colorby}.{format_save}'
    fn_save_init = f'{plot_dir}/{data_tag}_N{N_sub_str}_{property_colorby}_3d.png'

    if N_sub_str=='all':
        alpha = 0.1
        s = 0.11
    else:
        # to make visible for tests
        alpha = 1
        s = 4
    lim = 3000
    vmin, vmax = 0, 4.5

    print("Reading data:", fn_data)
    tab = Table.read(fn_data, format='fits')
    print("Loaded data with N =", len(tab))

    tab = prepare_data(tab, redshift_name, ra_name=ra_name, dec_name=dec_name, 
                        N_sub_str=N_sub_str)

    scmap = utils.shiftedColorMap(matplotlib.cm.plasma_r, start=0.2, midpoint=0.6, stop=1.0, name='plasma_shifted')
    # Create an init function and the animate functions.
    print(f"s = {s}, alpha={alpha}, lim={lim}, vmin={vmin}, vmax={vmax}")

    plot_init(tab, tab[property_colorby], s, alpha, lim, vmin, vmax,
                 cmap=scmap, colorbar_label=colorbar_label,
                 fn_save_init=fn_save_init)

    anim = make_anim(tab, tab[property_colorby], s, alpha, lim, vmin, vmax,
                     cmap=scmap, title=title, colorbar_label=colorbar_label)

    print("Saving animation to", fn_save)
    s = time.time()
    if format_save=='gif':
        anim.save(fn_save, writer=PillowWriter(fps=32)) 
                  #savefig_kwargs={'bbox_inches': 'tight'})
    elif format_save=='mp4':
        anim.save(fn_save, fps=30, extra_args=['-vcodec', 'libx264'])
    e = time.time()
    print("Save time:", e-s)
    
    print("Saved!")


def prepare_data(tab, redshift_name, ra_name='ra', dec_name='dec', 
                 N_sub_str='all', add_M=False):
    tab = tab.copy()

    # Subsample, if desired
    if N_sub_str!='all':
        N_sub = int(float(N_sub_str))
        tab = subsample(tab, N_sub)
        print("Subsampled data to N =", len(tab))

    # Convert to cartesian
    print("Converting to cartesian")
    cosmo = astropy.cosmology.Planck15
    add_xyz(tab, cosmo, redshift_name, ra_name=ra_name, dec_name=dec_name)

    # add properties
    print("Adding properties")
    #add_g_rp_color(data)
    if add_M:
        add_M_absolute(tab, cosmo, redshift_name)
    return tab


def plot_init(tab, c, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
                  title=None, colorbar_label=r'redshift $z$', colorbar=True,
                  fn_save_init=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    init_animation(fig, ax, tab, c, s, alpha, lim, vmin, vmax, cmap=cmap, 
                   title=title, colorbar_label=colorbar_label, colorbar=colorbar,
                   #fn_save_init=fn_save_init,
                   )
    if fn_save_init is not None:
       plt.savefig(fn_save_init, bbox_inches='tight')


def init_animation(fig, ax, tab, c, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
                  title=None, colorbar_label=r'redshift $z$', colorbar=True,
                  fn_save_init=None):
    #fig = plt.figure(figsize=(10,10))
    #ax = fig.add_subplot(projection='3d')
    ax.scatter(tab['x'], tab['y'], tab['z'], c=c, s=s, 
                      alpha=alpha, 
                      cmap=cmap, vmin=vmin, vmax=vmax
                      )
    ax.axis('off')

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_facecolor('white')
    if title is not None:
        ax.set_title(title, y=1.03, fontsize=28)

    # colorbar
    if colorbar:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cc = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        #cbaxes = fig.add_axes([1.05, 0.26, 0.02, 0.5]) # side
        cbaxes = fig.add_axes([0.92, 0.26, 0.02, 0.5]) # side, closer in
        cbar = fig.colorbar(cc, cax=cbaxes, orientation='vertical')
        #cbaxes = fig.add_axes([0.26, 0.09, 0.5, 0.02]) # bottom
        cbar.set_label(label=colorbar_label, fontsize=25)
        cbar.ax.tick_params(labelsize=22)
        #cbar.ax.set_yticklabels([0,1,2,3,4])
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

    if fn_save_init is not None:
        plt.savefig(fn_save_init, bbox_inches='tight')

    #return fig,


def make_anim(data, c, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
              title=None, colorbar_label=r'redshift $z$', colorbar=True):
    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    #ax = Axes3D(fig)

    def init():
        #fig = plt.figure(figsize=(10,10))
        #ax = fig.add_subplot(projection='3d')
        ax.scatter(data['x'], data['y'], data['z'], c=c, s=s, alpha=alpha, 
                          cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        #ax.set_facecolor('black')
        ax.set_facecolor('white')
        if title is not None:
            ax.set_title(title, y=1.03, fontsize=28)

        # colorbar
        if colorbar:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cc = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbaxes = fig.add_axes([1.05, 0.26, 0.02, 0.5]) # side
            cbar = fig.colorbar(cc, cax=cbaxes, orientation='vertical')
            #cbaxes = fig.add_axes([0.26, 0.09, 0.5, 0.02]) # bottom
            #cbar = fig.colorbar(cc, cax=cbaxes, orientation='horizontal')
            #cbaxes = fig.add_axes([0.8, 0.26, 0.02, 0.5]) 
            #cbaxes = fig.add_axes([0.5, 0.5, 0.01, 0.43]) 
            #cbar = fig.colorbar(cc, label=r'redshift $z$', extend='max', fraction=0.033, pad=0.06)
            cbar.set_label(label=colorbar_label, fontsize=25)
            cbar.ax.tick_params(labelsize=22)
            #cbar.ax.set_yticklabels([0,1,2,3,4])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    return anim


def add_xyz(data, cosmo, redshift_name, ra_name='ra', dec_name='dec'):
    dist_photoz = (cosmo.comoving_distance(data[redshift_name])*cosmo.h).value # convert to Mpc/h
    data.add_column(dist_photoz, name='distance')

    x, y, z = radec_to_cartesian(data['distance'], data[ra_name], data[dec_name])
    data.add_column(x, name='x')
    data.add_column(y, name='y')
    data.add_column(z, name='z')


def add_g_rp_color(data):
    g_rp = np.array(data['phot_g_mean_mag']) - np.array(data['phot_rp_mean_mag'])
    data.add_column(g_rp, name='g_rp')


def add_M_absolute(data, cosmo, redshift_name):
    dist_for_M = (10*u.pc).to(u.Mpc)
    distance_Mpc = data['distance']/cosmo.h
    m_apparent = np.array(data['phot_g_mean_mag'])
    D_luminosity = (1+data[redshift_name])*distance_Mpc*u.Mpc
    M_absolute = m_apparent - 5*np.log10(D_luminosity/dist_for_M)
    data.add_column(M_absolute, name='M_absolute_g')


def subsample(data, N_sub):
    N = len(data)
    idx_sub = np.random.choice(np.arange(N), size=N_sub, replace=False)
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
