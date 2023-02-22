import numpy as np
import pymaster as nmt
import healpy as hp

def make_counts(nside,l,b,weight=None):
    counts=np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside,l,b,lonlat=True)
    if weight is None:
        for p in pix:
            counts[p]+=1    
    else:
        i=0
        for p in pix:
            counts[p]+=weight[i]
            i+=1
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

def compute_master(f_a, f_b, wsp):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def compute_master_crosscorr_mask(klr,c,c1,c2,jk,apodized_mask,binning,lmax,cls_gg_th,cls_kg_th):
    
    nside = hp.npix2nside(len(klr))
    klm = hp.map2alm(klr,iter=1,pol=False)
    # corrects for pixel window function
    beam = hp.pixwin(nside,lmax=lmax,pol=False)
    
    f0 = nmt.NmtField(apodized_mask, [klr],beam=beam) # corrects for pixel window as klr computed from downgrade

    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0, f0, binning)    

    f1 = nmt.NmtField( apodized_mask, [c-c[apodized_mask!=0].mean()],beam=beam)
    f11 = nmt.NmtField(apodized_mask, [c1],beam=beam)
    f12 = nmt.NmtField(apodized_mask, [c2],beam=beam)
    fjk = nmt.NmtField(apodized_mask, [jk],beam=beam)
    
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
