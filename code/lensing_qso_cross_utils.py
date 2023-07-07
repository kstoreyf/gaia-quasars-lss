import numpy as np
import pymaster as nmt
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord

def make_counts(nside,l,b,weight=None,mean_counts=False):
    counts=np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside,l,b,lonlat=True)
    if weight is None:
        for p in pix:
            counts[p]+=1    
    else:
        nobs = np.zeros_like(counts)
        i=0
        for p in pix:
            counts[p]+=weight[i]
            i+=1
            nobs[p]+=1
        if mean_counts:
            counts[counts!=0]/=nobs[counts!=0]
        #print(np.sum(counts),np.sum(nobs))
    return counts

def define_binning(lmin,lmax,delta_b,nside,weighting='ivar'):
    if weighting=='ivar':
        if ((type(lmin) is np.ndarray) and (type(lmax) is np.ndarray)):
            ells = np.arange(lmax[-1], dtype='int32')
        else:
            ells = np.arange(lmax+1, dtype='int32')
        weights = (2*ells+1.)
        bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
        if ((type(lmin) is np.ndarray) and (type(lmax) is np.ndarray)):
             for i,(lmin_i,lmax_i) in enumerate(zip(lmin,lmax)):
                 bpws[lmin_i:lmax_i+1]=i
        else:
            #ells = np.arange(lmax+1, dtype='int32')
            #weights = (2*ells+1.)
            #bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
            i = 0
            while delta_b * (i + 1) + lmin < lmax+1:
                bpws[delta_b * i + lmin:delta_b * (i + 1) + lmin] = i
                i += 1
        b = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights)
    elif weighting=='flat':
        if ((type(lmin) is np.ndarray) and (type(lmax) is np.ndarray)):
            b = nmt.NmtBin.from_edges(lmin,lmax,is_Dell=False) 
        else:
            ell_bounds=np.arange(lmin,lmax,delta_b)
            b= nmt.NmtBin.from_edges(ell_bounds[:-1], ell_bounds[1:], is_Dell=False)
    else:
        # assumes lmax=3nside-1
        b = nmt.NmtBin.from_lmax_linear(lmax, delta_b,is_Dell=False)

    return b

def get_custom_binning(delta_b=20,delta_b_high_l=50,lmax_transition=100,nside=256,lmax=767,weighting='ivar'):
    lmin_low_ells = np.arange(0,lmax_transition,delta_b)
    lmax_low_ells = lmin_low_ells + delta_b
    lmin_low_ells[0]=2
    lmin_high_ells = np.arange(lmax_transition,lmax-delta_b_high_l,delta_b_high_l)
    lmax_high_ells = lmin_high_ells+delta_b_high_l
    lmax_high_ells[-1]=lmax
    if delta_b==2:
        # removes first 2 multipoles
        lmin_low_ells = lmin_low_ells[1:]
        lmax_low_ells = lmax_low_ells[1:]
    lmins = np.append(lmin_low_ells,lmin_high_ells)
    lmaxs = np.append(lmax_low_ells,lmax_high_ells)
    
    binning = define_binning(lmins,lmaxs,delta_b=delta_b,nside=nside,weighting=weighting)
    return binning


def compute_master(f_a, f_b, wsp):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def compute_master_crosscorr_mask(klr,c,c1,c2,jk,apodized_mask,binning,lmax,cls_gg_th=None,cls_kg_th=None,gsyst=None,return_mode_coupling=False,w=None,beam_k=True,beam_g=True):
    
    nside = hp.npix2nside(len(klr))
    # corrects for pixel window function
    if beam_k:
        beam_k = hp.pixwin(nside,lmax=lmax,pol=False)
    else:
        beam_k = None
    if beam_g:
        beam_g = hp.pixwin(nside,lmax=lmax,pol=False)
    else:
        beam_g = None        
    
    f0 = nmt.NmtField(apodized_mask, [klr],beam=beam_k) # corrects for pixel window as klr computed from downgrade
    if w is None:
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f0, f0, binning)    

    f1 = nmt.NmtField( apodized_mask, [c-c[apodized_mask!=0].mean()],beam=beam_g,templates=gsyst)
    f11 = nmt.NmtField(apodized_mask, [c1],beam=beam_g,templates=gsyst)
    f12 = nmt.NmtField(apodized_mask, [c2],beam=beam_g,templates=gsyst)
    fjk = nmt.NmtField(apodized_mask, [jk],beam=beam_g,templates=gsyst)
    
    clkg = compute_master(f0, f1, w)[0]
    clgg = compute_master(f1, f1, w)[0]
    clkk = compute_master(f0, f0, w)[0]
    
    clkg1 = compute_master(f0, f11, w)[0]
    clg1g1 = compute_master(f11, f11, w)[0]
    
    clkg2 = compute_master(f0, f12, w)[0]
    clg2g2 = compute_master(f12, f12, w)[0]
    
    
    clg1g2 = compute_master(f11, f12, w)[0]
    clkgjk = compute_master(f0, fjk, w)[0]
    clgjk = compute_master(fjk, fjk, w)[0]
    if ((cls_gg_th is None) and (cls_kg_th is None)):
        if return_mode_coupling:
            return clkg,clgg,clkk,clkg1,clg1g1,clkg2,clg2g2,clg1g2,clkgjk,clgjk,w
        else:
            return clkg,clgg,clkk,clkg1,clg1g1,clkg2,clg2g2,clg1g2,clkgjk,clgjk
    else:
        klm = hp.map2alm(klr,iter=1,pol=False)
        cl_gg_th_binned = w.decouple_cell(w.couple_cell([cls_gg_th]))[0]
        cl_kg_th_binned = w.decouple_cell(w.couple_cell([cls_kg_th]))[0]
        cl_kk_th_binned = w.decouple_cell(w.couple_cell([hp.alm2cl(klm)]))[0]    
        return clkg,clgg,clkk,clkg1,clg1g1,clkg2,clg2g2,clg1g2,clkgjk,clgjk,cl_gg_th_binned,cl_kg_th_binned,cl_kk_th_binned


def overdensity_from_counts(m,footprint,verbose=False):
    if type(footprint) is np.ndarray:
        nbar=np.mean(m[footprint])
    else:
        nbar = float(footprint)
    if verbose:
        print("nbar",nbar)
    c = m/nbar -1
    return c

def get_magellanic_cloud_mask(nside,r_mclouds=[4,2]):
    # Magellanic Cloud mask
    # Give in input nside and radius for masking in degrees
    mclouds = [(280.4652,-32.8884),(302.8084,-44.3277)] #Large MC and Small MC coordinates
    #r_mclouds = [4,2] #deg
    #r_mclouds = [5,2] #deg
    mclouds_mask = np.ones(hp.nside2npix(nside))
    for i,lmc in enumerate(mclouds):
        mcpix = hp.query_disc(nside,hp.ang2vec(lmc[0],lmc[1],lonlat=True),np.deg2rad(r_mclouds[i]))
        mclouds_mask[mcpix]=0.
    return mclouds_mask

def extrapolpixwin(nside, Slmax, pixwin=True):
    '''
    Parameters
    ----------
    nside : int 
        Healpix map resolution
    Slmax : int 
        Maximum multipole value computed for the pixel covariance pixel matrix
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
        
    Returns
    ----------
    fpixwin : array of floats

    Example :
    ----------
    >>> print(hp.pixwin(2))
    [ 1.          0.977303    0.93310702  0.86971852  0.79038278  0.69905215
      0.60011811  0.49813949  0.39760902]
    >>> print(extrapolpixwin(2, 20, True)) 
    [  1.00000000e+00   9.77303000e-01   9.33107017e-01   8.69718524e-01
       7.90382779e-01   6.99052151e-01   6.00118114e-01   4.98139486e-01
       3.97609016e-01   3.07026358e-01   2.27432772e-01   1.61472532e-01
       1.09618639e-01   7.09875545e-02   4.37485835e-02   2.55977424e-02
       1.41862346e-02   7.42903370e-03   3.66749182e-03   1.70274467e-03
       7.41729191e-04]
    >>> print(extrapolpixwin(2, 20, False))
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.  1.]
    '''
    if pixwin:
        prepixwin = np.array(hp.pixwin(nside))
        poly = np.polyfit(np.arange(len(prepixwin)), np.log(prepixwin),
                          deg=3, w=np.sqrt(prepixwin))
        y_int = np.polyval(poly, np.arange(Slmax+1))
        fpixwin = np.exp(y_int)
        fpixwin = np.append(prepixwin, fpixwin[len(prepixwin):])[:Slmax+1]
    else:
        fpixwin = np.ones((Slmax+1))

    return fpixwin

def prepare_gaia_catalog(gaia_fname,verbose=False,snr_zcut=0.):# Prepare raw GAIA data
    if verbose:
        print("Read catalog and convert coordinate")
    d=fits.open(gaia_fname)

    z = d[1].data['redshift_spz']
    try:
        zerr = d[1].data['redshift_spz_err']
        snr = z/zerr
    except:
        # works for old routines
        zerr=np.zeros(len(z))
        snr = 3*np.ones(len(z))
    sc = SkyCoord(ra=d[1].data["ra"], dec=d[1].data["dec"], unit='deg', frame='icrs', equinox='J2000.0')
    gs = sc.transform_to(frame='galactic')
    l = gs.l.value[snr>snr_zcut] 
    b = gs.b.value[snr>snr_zcut] 
    nqso = len(l)
    return z[snr>snr_zcut] ,zerr[snr>snr_zcut] ,l,b,nqso

