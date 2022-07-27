import time

from Corrfunc.theory import DD
from Corrfunc.theory import xi
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.utils import convert_rp_pi_counts_to_wp



def compute_wtheta(theta_edges, ra, dec, ra_rand, dec_rand,
                   return_full_results=False, nthreads=4):
        
    autocorr = 1
    start = time.time()
    DD_theta = DDtheta_mocks(autocorr, nthreads, theta_edges, ra, dec)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    autocorr = 0
    start = time.time()
    DR_theta = DDtheta_mocks(autocorr, nthreads, theta_edges,
                               ra, dec,
                               RA2=ra_rand, DEC2=dec_rand)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    start = time.time()
    autocorr = 1
    RR_theta = DDtheta_mocks(autocorr, nthreads, theta_edges, ra_rand, dec_rand)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    N = len(ra)
    N_rand = len(ra_rand)
    wtheta = convert_3d_counts_to_cf(N, N, N_rand, N_rand,
                                 DD_theta, DR_theta,
                                 DR_theta, RR_theta)
    
    if return_full_results:
        return wtheta, DD_theta, DR_theta, RR_theta
    
    return wtheta


def compute_xi(r_edges, x, y, z, x_rand, y_rand, z_rand,
               return_full_results=False):
    
    nthreads = 4
    
    start = time.time()
    autocorr=1
    res_dd = DD(autocorr, nthreads, r_edges, x, y, z, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    
    start = time.time()
    autocorr=0
    res_dr = DD(autocorr, nthreads, r_edges, x, y, z,
                X2=x_rand, Y2=y_rand, Z2=z_rand, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')

    start = time.time()
    autocorr=1
    res_rr = DD(autocorr, nthreads, r_edges, x_rand, y_rand, z_rand, periodic=False)
    end = time.time()
    print(f'Time: {end-start:.4f} s')
    print()
    
    N = len(x)
    N_rand = len(x_rand)
    xi = convert_3d_counts_to_cf(N, N, N_rand, N_rand,
                                     res_dd, res_dr,
                                     res_dr, res_rr)
    
    if return_full_results:
        return xi, res_dd, res_dr, res_rr
    
    return xi