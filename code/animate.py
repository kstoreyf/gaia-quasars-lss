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
matplotlib.rc('text', usetex=True)

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
    #data_tag = '2panel'
    tag_cat = ''

    #plot_dir = '../plots/2023-04-04_figures'
    anim_dir = '../plots/2023-05-27_animations'
    plot_dir = anim_dir
    make_image = True
    make_anim = True
    #tag_anim = '_alpha0.1_black'
    tag_anim = '_cbar_setazim'

    if data_tag=='2panel':
        data_arr = []

    if 'gcat' in data_tag:
        fn_data = f'../data/quaia_G{G_max}{tag_cat}.fits'
        #title = f'Quaia ($G<{G_max}$)'
        title = None
        colorbar_label = r'redshift $z$'
        redshift_name = 'redshift_quaia'
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

    # mp4 works, unsure about gif rn
    #format_save = 'gif'
    format_save = 'mp4'
    #N_sub_str = '1e3'
    N_sub_str = 'all'
    
    fn_save = f'{anim_dir}/animation_{data_tag}_N{N_sub_str}_{property_colorby}{tag_anim}.{format_save}'
    fn_save_init = f'{plot_dir}/image_{data_tag}_N{N_sub_str}_{property_colorby}{tag_anim}_3d.png'

    if N_sub_str=='all':
        #alpha = 0.1
        #s = 0.11
        alpha = 0.03
        s = 0.1
        # alpha = 0.1
        # s = 0.03
    else:
        # to make visible for tests
        alpha = 1
        s = 4
    lim = 3150
    vmin, vmax = 0, 4.5
    facecolor = 'white'
    #facecolor = 'black'

    print("Reading data:", fn_data)
    tab = Table.read(fn_data, format='fits')
    print("Loaded data with N =", len(tab))

    tab = prepare_data(tab, redshift_name, ra_name=ra_name, dec_name=dec_name, 
                        N_sub_str=N_sub_str)

    if facecolor=='black':
        scmap = utils.shiftedColorMap(matplotlib.cm.plasma, start=0.2, midpoint=0.6, stop=1.0, name='plasma_shifted')
    else:
        # white is default
        scmap = utils.shiftedColorMap(matplotlib.cm.plasma_r, start=0.2, midpoint=0.6, stop=1.0, name='plasma_r_shifted')
    
    # Create an init function and the animate functions.
    print(f"s = {s}, alpha={alpha}, lim={lim}, vmin={vmin}, vmax={vmax}")

    if make_image:
        plot_init(tab, tab[property_colorby], s, alpha, lim, vmin, vmax,
                    cmap=scmap, colorbar_label=colorbar_label,
                    fn_save_init=fn_save_init, facecolor=facecolor)
        print("Saved image!")

    if make_anim:
        anim = make_animation(tab, tab[property_colorby], s, alpha, lim, vmin, vmax,
                        cmap=scmap, title=title, colorbar_label=colorbar_label,
                        facecolor=facecolor)

        print("Saving animation to", fn_save)
        s = time.time()
        if format_save=='gif':
            anim.save(fn_save, writer=PillowWriter(fps=32), 
                      )
        elif format_save=='mp4':
            anim.save(fn_save, fps=30, extra_args=['-vcodec', 'libx264'],
                      )
        e = time.time()
        print("Save time:", e-s)
    
   

def main_2panel():
    #fn_data = '../data/gaia_photoz.fits'
    #fn_data = '/scratch/ksf293/gaia-quasars-lss/data/gaia_photoz.fits'
    #fn_data = '/scratch/ksf293/gaia-quasars-lss/data/gaia_wise_panstarrs_tmass.fits.gz'
    G_max = 20.5
    #data_tag = f'gcathi'
    #data_tag = 'sdss'
    data_tags = ['gcathi', 'sdss']
    data_tag = '_'.join(data_tags)

    tag_cat = ''

    #plot_dir = '../plots/2023-04-04_figures'
    anim_dir = '../plots/2023-05-27_animations'
    plot_dir = anim_dir
    image_only = True
    #make_anim = True
    #tag_anim = '_alpha0.1_black'
    tag_anim = '_cbar_setazim_alpha0.07'

    fn_datas, titles, redshift_names, properties_colorby, radec_names = [], [], [], [], []
    if 'gcathi' in data_tags:
        fn_datas.append( f'../data/quaia_G{G_max}{tag_cat}.fits' )
        titles.append( f'Quaia' )
        #titles.append( f'Quaia ($G<{G_max}$)' )
        #titles.append( None )
        redshift_names.append( 'redshift_quaia' )
        properties_colorby.append( 'redshift_quaia' )
        radec_names.append( ('ra', 'dec') )
    if 'sdss' in data_tags:
        fn_datas.append( f'../data/quasars_sdss_xgaiaall_xunwiseall_good.fits' )
        #titles.append( None )
        titles.append( 'SDSS DR16Q' )
        redshift_names.append( 'z_sdss' )
        properties_colorby.append( 'z_sdss' )
        radec_names.append( ('ra_sdss', 'dec_sdss') )

    assert len(data_tags)==2, "Wrong number of datasets!"

    # mp4 works, unsure about gif rn
    #format_save = 'gif'
    format_save = 'mp4'
    #N_sub_str = '1e3'
    N_sub_str = 'all'
    
    fn_save = f'{anim_dir}/animation_{data_tag}_N{N_sub_str}{tag_anim}.{format_save}'
    fn_save_init = f'{plot_dir}/image_{data_tag}_N{N_sub_str}{tag_anim}_3d.png'

    if N_sub_str=='all':
        #alpha = 0.1
        #s = 0.11
        alpha = 0.03
        s = 0.1
        # alpha = 0.1
        # s = 0.03
        if image_only:
            # for some reason it looks diff in image and vid! so make it look more similar
            alpha = 0.07
            s = 0.1
    else:
        # to make visible for tests
        alpha = 1
        s = 4
    lim = 3250
    vmin, vmax = 0, 4.5
    facecolor = 'white'
    #facecolor = 'black'

    if facecolor=='black':
        scmap = utils.shiftedColorMap(matplotlib.cm.plasma, start=0.2, midpoint=0.6, stop=1.0, name='plasma_shifted')
    else:
        # white is default
        scmap = utils.shiftedColorMap(matplotlib.cm.plasma_r, start=0.2, midpoint=0.6, stop=1.0, name='plasma_r_shifted')

    data_arr = []
    for j, fn_data in enumerate(fn_datas):
        print("Reading data:", fn_data)
        tab = Table.read(fn_data, format='fits')
        print("Loaded data with N =", len(tab))

        tab = prepare_data(tab, redshift_names[j], ra_name=radec_names[j][0], dec_name=radec_names[j][1], 
                            N_sub_str=N_sub_str)
        data_arr.append(tab)
        
    # Create an init function and the animate functions.
    print(f"s = {s}, alpha={alpha}, lim={lim}, vmin={vmin}, vmax={vmax}")

    # if make_image:
    #     plot_init(tab, tab[property_colorby], s, alpha, lim, vmin, vmax,
    #                 cmap=scmap, colorbar_label=colorbar_label,
    #                 fn_save_init=fn_save_init, facecolor=facecolor)
    #     print("Saved image!")

    #if make_anim:
    vals_colorby = [data_arr[j][properties_colorby[j]] for j in range(len(data_arr))]
    anim = make_animation_2panel(data_arr, vals_colorby, s, alpha, lim, vmin, vmax,
                    cmap=scmap, title_arr=titles,
                    facecolor=facecolor, fn_save_init=fn_save_init)

    if not image_only:
        print("Saving animation to", fn_save)
        s = time.time()
        if format_save=='gif':
            anim.save(fn_save, writer=PillowWriter(fps=32), 
                      )
        elif format_save=='mp4':
            anim.save(fn_save, fps=30, extra_args=['-vcodec', 'libx264'],
                      )
        e = time.time()
        print("Save time:", e-s)
    


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
                  fn_save_init=None, facecolor='white'):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    init_animation(fig, ax, tab, c, s, alpha, lim, vmin, vmax, cmap=cmap, 
                   title=title, colorbar_label=colorbar_label, colorbar=colorbar,
                   facecolor=facecolor
                   #fn_save_init=fn_save_init,
                   )
    if fn_save_init is not None:
       plt.savefig(fn_save_init, bbox_inches='tight')


def init_animation(fig, ax, tab, c, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
                  title=None, colorbar_label=r'redshift $z$', colorbar=True,
                  fn_save_init=None, facecolor='white'):
    #fig = plt.figure(figsize=(10,10))
    #ax = fig.add_subplot(projection='3d')
    ax.scatter(tab['x'], tab['y'], tab['z'], c=c, s=s, 
                      alpha=alpha, 
                      cmap=cmap, vmin=vmin, vmax=vmax
                      )
    ax.axis('off')
    ax.view_init(elev=10., azim=-80)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_facecolor(facecolor)
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


def make_animation(data, c, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
              title=None, colorbar_label=r'redshift $z$', colorbar=True,
              facecolor='white', elev=10., azim=-80.):
    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    #fig.subplots_adjust(left=0, bottom=0, right=0.83, top=1, wspace=None, hspace=None)
    fig.subplots_adjust(left=0, bottom=0, right=0.85, top=1, wspace=None, hspace=None)
    ax.view_init(elev=elev, azim=azim)

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
        ax.set_facecolor(facecolor)
        if title is not None:
            ax.set_title(title, y=1.03, fontsize=28)

        # colorbar
        if colorbar:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cc = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            #cbaxes = fig.add_axes([1.05, 0.26, 0.02, 0.5]) # side
            #cbaxes = fig.add_axes([0.96, 0.26, 0.02, 0.5]) # side, middle in
            #cbaxes = fig.add_axes([0.92, 0.26, 0.02, 0.5]) # side, closer in
            cbaxes = fig.add_axes([0.89, 0.26, 0.02, 0.5]) # side, even closer in
            #cbaxes.patch.set_alpha(0.0)
            cbar = fig.colorbar(cc, cax=cbaxes, orientation='vertical')
            #cbaxes = fig.add_axes([0.26, 0.09, 0.5, 0.02]) # bottom
            #cbar = fig.colorbar(cc, cax=cbaxes, orientation='horizontal')
            #cbaxes = fig.add_axes([0.8, 0.26, 0.02, 0.5]) 
            #cbaxes = fig.add_axes([0.5, 0.5, 0.01, 0.43]) 
            #cbar = fig.colorbar(cc, label=r'redshift $z$', extend='max', fraction=0.033, pad=0.06)
            cbar.set_label(label=colorbar_label, fontsize=25)
            cbar.ax.tick_params(labelsize=22)
            #cbar.ax.zorder = -1
            #cbar.ax.set_yticklabels([0,1,2,3,4])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

        return fig,

    def animate(i):
        ax.view_init(elev=elev, azim=azim+i)
        #ax.view_init(azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    return anim


def make_animation_2panel(data_arr, c_arr, s, alpha, lim, vmin, vmax, cmap='plasma_r', 
              title_arr=None, colorbar_label=r'redshift $z$', colorbar=True,
              facecolor='white', elev=10., azim=-80., fn_save_init=None):
    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(20,8))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')
    ax_arr = [ax0, ax1]
    fig.subplots_adjust(left=0, bottom=-0.05, right=0.9, top=0.95, wspace=-0.1, hspace=None)
    ax0.view_init(elev=elev, azim=azim)
    ax1.view_init(elev=elev, azim=azim)

    #ax = Axes3D(fig)

    def init():
        #fig = plt.figure(figsize=(10,10))
        #ax = fig.add_subplot(projection='3d')
        for j, ax in enumerate(ax_arr):
            data = data_arr[j]
            ax.scatter(data['x'], data['y'], data['z'], c=c_arr[j], s=s, alpha=alpha, 
                            cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_facecolor(facecolor)
            if title_arr is not None:
                ax.set_title(title_arr[j], y=1.03, fontsize=28)

        # colorbar
        if colorbar:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cc = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            #cbaxes = fig.add_axes([1.05, 0.26, 0.02, 0.5]) # side
            #cbaxes = fig.add_axes([0.92, 0.26, 0.02, 0.5]) # side, closer in
            cbaxes = fig.add_axes([0.92, 0.2, 0.012, 0.5]) # side, closer in, shorter
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
        ax0.view_init(elev=elev, azim=azim+i)
        ax1.view_init(elev=elev, azim=azim+i)
        return fig,

    # First save initial frame as image
    if fn_save_init is not None:
        init()
        plt.savefig(fn_save_init)
        print("Saved initial frame as image to", fn_save_init)

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
    #main()
    main_2panel()
