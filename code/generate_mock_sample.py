import numpy as np
import struct
from numpy.random import default_rng

import astropy
import astropy.cosmology
from astropy.table import Table
from astropy import units as u
from scipy.integrate import quad
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

import utils


def main():

    # save name
    m_apparent_lim = 20
    tag_mock = f'_maglim{m_apparent_lim}'
    fn_sample = f'../data/randoms/sample_mock{tag_mock}.fits'
    overwrite = True

    # general settings
    rng = default_rng(seed=42)
    cosmo = astropy.cosmology.Planck18
    
    # Load and set up data
    L, N_full, data = load_stitched_mock()
    print(f"Initial number of sources: {N_full}")

    # SUBSAMPLE FOR TESTING CODE ONLY
    #data = data[rng.integers(low=0, high=N_full, size=1000)]
    #print(f"Testing sources: {len(data)}")

    # Start building table
    tab = make_initial_sample_table(data, L, cosmo)
    add_velocity_effects(tab, cosmo, rng, rsd=True, smear=True)

    add_magnitudes(tab, cosmo)
    tab = apparent_mag_lim(tab, m_apparent_lim)
    print(f"Number of sources after magnitude cut to {m_apparent_lim}: {len(tab)}")

    add_ra_dec(tab)

    # Save! 
    tab.write(fn_sample, overwrite=overwrite)
    print(f"Wrote sample to {fn_sample}!")


def make_initial_sample_table(data, L, cosmo):
    print("Constructing table")
    # Shift the box to be centered on zero
    pos = data[:,:3]
    pos -= L/2 # do this in-place to save memory; want 0 at center of box
    vel = data[:,3:6]

    print("Computing distances and redshifts")
    z_max = 8
    distances = np.linalg.norm(pos, axis=1)
    redshifts = distances_to_redshifts(distances, cosmo, z_max_manual=z_max)

    # ones greater than zmax won't get redshifts; just delete them now
    idx_finite = np.isfinite(redshifts)

    col_names = ['x_true', 'y_true', 'z_true', 'vx_true', 'vy_true', 'vz_true',
                 'dist_true', 'redshift_true']
    data_cols = np.vstack([pos.T, vel.T, distances, redshifts]).T
    data_cols = data_cols[idx_finite]
    tab = Table(data_cols, names=col_names)
    print(f"Constructed table with {len(tab)} sources (initial: {len(data)})")
    return tab



def read_lognormal_mock(fn):

    with open(fn, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        
        nleading = 3*8+1*4
        header = struct.unpack("dddi", fileContent[:nleading])
        Lx, Ly, Lz, N = header
        data = struct.unpack("f" * ((len(fileContent) -nleading) // 4), fileContent[nleading:])
         
    data = np.array(data)
    data = data.reshape((-1, 6)) 
        
    return Lx, Ly, Lz, N, data


def load_single_mock(rlz):
    fn_mock = f'/content/drive/My Drive/lognormal_mocks_quasars/cat_L6000_N2e6_z0_patchy_lognormal_rlz{rlz}.bin'
    Lx, Ly, Lz, N, data = read_lognormal_mock(fn_mock)
    assert Lx==Ly and Lx==Lz, "Box not cubic!"
    L = Lx
    return L, N, data


def load_stitched_mock():
    print("Loading in stitched mock")
    data = []
    count = 0
    N_full = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                fn_mock = f'../data/mocks/cat_L6000_N2e6_z0_patchy_lognormal_rlz{count}.bin'
                Lx, Ly, Lz, N, data_single = read_lognormal_mock(fn_mock)
                assert Lx==Ly and Lx==Lz, "Box not cubic!"
                L = Lx
                N_full += N
                shifts = np.array([i*L, j*L, k*L])
                data_single[:,:3] += shifts # shift positions
                data.extend(data_single)
                count += 1

    data = np.array(data)
    L_full = Lx*2
    return L_full, N_full, data


### Redshift & velocity effects functions

def distances_to_redshifts(distances_Mpcperh, cosmo, z_max_manual=8):
    print("Converting distances to redshifts")
    # distances are in units of Mpc/h; need units of Mpc for the astropy function
    distances_Mpc = distances_Mpcperh / cosmo.h 
    # z_at_value function takes a long time, so we interpolate, following
    # https://docs.astropy.org/en/stable/api/astropy.cosmology.z_at_value.html#astropy.cosmology.z_at_value

    z_min = astropy.cosmology.z_at_value(cosmo.comoving_distance, distances_Mpc[~np.isnan(distances_Mpc)].min() * u.Mpc)
    distance_max = min( cosmo.comoving_distance(z_max_manual).value, distances_Mpc[~np.isnan(distances_Mpc)].max() )
    z_max = astropy.cosmology.z_at_value(cosmo.comoving_distance, distance_max * u.Mpc)

    z_grid = np.linspace(z_min, z_max, 100)
    dist_grid = cosmo.comoving_distance(z_grid)
    z_vals = np.interp(distances_Mpc, dist_grid.value, z_grid, left=np.nan, right=np.nan)
    return z_vals.value


def position_shift_rsd(vel, redshifts, cosmo):
    # http://mwhite.berkeley.edu/Talks/SantaFe12_RSD.pdf
    # i think the velocities given by lognormal_galaxies are peculiar velocites, 
    # as i'd been assuming, bc of this: (https://arxiv.org/pdf/1706.09195.pdf)
    # "To ensure that the galaxy overdensities and velocities are correlated, 
    # we use the same random seed for G of galaxies and matter."

    # CHECK LITTLE H
    return ( ((1+redshifts)*vel.T) / cosmo.H(redshifts).value ).T


def position_shift_smear(rng, pos, redshifts, cosmo):
    smear_mean = 0 
    smear_stdev = 500 #km/s
    vel_smear_mag = rng.normal(loc=smear_mean, scale=smear_stdev, size=pos.shape[0])
    _, theta, phi = utils.cartesian_to_spherical(*pos.T)
    vel_smear_components = utils.spherical_to_cartesian(vel_smear_mag, theta, phi)
    return ( ((1+redshifts)*vel_smear_components) / cosmo.H(redshifts).value ).T


def add_velocity_effects(tab, cosmo, rng, rsd=True, smear=True):
    print("Adding velocity effects")
    pos = np.array([tab['x_true'], tab['y_true'], tab['z_true']]).T
    vel = np.array([tab['vx_true'], tab['vy_true'], tab['vz_true']]).T
    redshifts = tab['redshift_true']

    pos_zspace = pos
    if rsd:
        pos_zspace += position_shift_rsd(vel, redshifts, cosmo)
    if smear:
        pos_zspace += position_shift_smear(rng, pos, redshifts, cosmo)

    distances_zspace = np.linalg.norm(pos_zspace, axis=1)
    redshifts_zspace = distances_to_redshifts(distances_zspace, cosmo, z_max_manual=7)

    # TODO: way to do this more efficiently?
    tab.add_column(redshifts_zspace, name='redshift_zspace')
    tab.add_column(distances_zspace, name='dist_zspace')
    tab.add_column(pos_zspace[:,0], name='x_zspace')
    tab.add_column(pos_zspace[:,1], name='y_zspace')
    tab.add_column(pos_zspace[:,2], name='z_zspace')


### QLF functions

# eqns 6-8, fixed version; https://arxiv.org/pdf/astro-ph/0601434.pdf
# Φ(M, z)dM is the number of quasars per unit comoving volume at redshift, 
# z, with absolute magnitudes between M −dM/2 and M +dM/2
def quasar_luminosity_function(M, z):
    A1 = 0.78
    B1 = 0.10
    B2 = 27.35
    B3 = 19.27
    M_star = -26
    z_ref = 2.45
    phi_star = 10**(-5.70)

    xi = np.log10((1 + z)/(1 + z_ref))
    mu = M - (M_star + B1*xi + B2*xi**2 + B3*xi**3)
    qlf = phi_star * 10**(A1*mu)
    return qlf


# define inverse so we can properly sample
def inverse_cdf_of_qlf(u, z, a, integral):

  A1 = 0.78
  B1 = 0.10
  B2 = 27.35
  B3 = 19.27
  M_star = -26
  z_ref = 2.45
  phi_star = 10**(-5.70)

  xi = np.log10((1 + z)/(1 + z_ref))
  beta = M_star + B1*xi + B2*xi**2 + B3*xi**3

  return 1/A1 * np.log10((u*A1*np.log(10))/(phi_star/integral) + 10**(a-beta)) + beta


def skewnorm_dist(x, a, amplitude, loc, scale):
  return amplitude*skewnorm.pdf(x, a, loc=loc, scale=scale)


def get_QLF_pdf_normalization(z_max, mag_min, mag_max):
    # First, we need the integral of the PDF for normalization, and it 
    # takes too long to do an every z, so we fit this
    z_fine = np.linspace(0, z_max, 1000)
    integrals = []
    for z_mid in z_fine:
        integral, _ = quad(quasar_luminosity_function, mag_max, mag_min, args=(z_mid))
        integrals.append(integral)
    popt_skew, _ = curve_fit(skewnorm_dist, z_fine, integrals)

    def QLF_pdf_normalization(redshifts):
        return skewnorm_dist(redshifts, *popt_skew)
    
    return QLF_pdf_normalization


def add_magnitudes(tab, cosmo, M_min=-23, M_max=-30):
    print("Adding magnitudes")
    # i think the magnitudes should be based on the sources' true redshifts/dists,
    # because that's what determines how bright they look, even if we measure dist wrong
    # TODO: check!
    redshifts = tab['redshift_true']
    distances_Mpc = utils.Mpcperh_to_Mpc(tab['dist_true'], cosmo)
    uniform = np.random.uniform(size=len(redshifts))
    z_max = np.max(redshifts)
    QLF_pdf_normalization = get_QLF_pdf_normalization(z_max, M_min, M_max) 
    norms = QLF_pdf_normalization(redshifts)
    M_absolute = inverse_cdf_of_qlf(uniform, redshifts, M_max, norms)

    dist_for_M = (10*u.pc).to(u.Mpc)

    # what distances should these be?? associated with these redshifts? 
    D_luminosity = (1+redshifts)*distances_Mpc*u.Mpc
    m_apparent = M_absolute + 5*np.log10(D_luminosity/dist_for_M)

    tab.add_column(M_absolute, name='M_absolute')
    tab.add_column(m_apparent.value, name='m_apparent')


def apparent_mag_lim(tab, m_apparent_lim):
    idx_maglim = np.where(tab['m_apparent'] < m_apparent_lim)[0]
    return tab[idx_maglim]


def add_ra_dec(tab):
    # zspace shouldn't effect ra and dec, so could use true, but might be numerical diffs
    pos = np.array([tab['x_zspace'], tab['y_zspace'], tab['z_zspace']])
    ra_zspace, dec_zspace = utils.cartesian_to_radec(*pos)
    tab.add_column(ra_zspace, name='ra')
    tab.add_column(dec_zspace, name='dec')


if __name__=='__main__':
    main()