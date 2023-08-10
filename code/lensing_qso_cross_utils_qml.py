import healpy as hp 
import numpy as np
from lensing_qso_cross_utils import make_counts, overdensity_from_counts, extrapolpixwin 
import pylab as pl 

def prepare_k_data(nside,pr4_alms='',pixwin = None,overwrite=False,lmax=None,remove_dipole=False):
    try:
        if overwrite:
            raise ValueError("Force recomputation of LR kappa maps")
        else:
            print("Read",in_map)
            klr=hp.read_map(in_map)
    except:
        klm = hp.read_alm(pr4_alms) # read DR4 instead of DR3      
        klm[0]=0.+0.j
        lmax = hp.Alm.getlmax(len(klm))
        if pixwin is None:
            print("Cut pixwin at lmax")
            fl=np.ones(lmax+1)
            lmax_pixwin = 3*nside if (lmax>=3*nside-1) else lmax+1
            #print(lmax+1,lmax_pixwin,3*nside-1)
            fl[:lmax_pixwin] = hp.pixwin(nside,lmax=lmax_pixwin-1)
            fl[lmax_pixwin:]=0.
            if remove_dipole:
                print("removing dipole from lensing alms")
                fl[1]=0.
            khr=hp.alm2map(hp.almxfl(klm,fl),nside=nside,pol=False)
            klr = hp.ud_grade(khr,nside_out=nside)            
            khr=0.                    
        else:
            pixwin_hr = extrapolpixwin(nside,lmax)
            fl = pixwin_hr
            if remove_dipole:
                print("removing dipole from lensing alms")
                fl[1]=0.            
            klr=hp.alm2map(hp.almxfl(klm,fl),nside=nside,lmax=lmax,pol=False)
        #hp.write_map(in_map,klr,coord='G',overwrite=overwrite)
    return klr


def counts_selcorrected(l,b,selection,nside,weight=None):
    selection_mask = selection>0
    m=make_counts(nside,l,b,weight=weight)
    msel=np.zeros(hp.nside2npix(nside))
    msel[selection_mask] = m[selection_mask]/selection[selection_mask]
    return msel

def process_catalog(l,b,selection,nside,nbar_confidence_mask=None,weight=None,verbose=False):
    #msel=counts_selcorrected(l,b,selection,nside,weight)
    #selection_mask = selection>0
    #if nbar_confidence_mask is None:
    #    nbar = np.mean(msel[selection_mask]) 
    #else:
    #    nbar = np.mean(msel[selection_mask&nbar_confidence_mask]) 
    #if verbose:
    #    print("nbar",nbar)
    #csel = overdensity_from_counts(msel,nbar,verbose=False) 

    msel=counts_selcorrected(l,b,np.ones_like(selection),nside,weight)
    selection_mask = selection>0
    if nbar_confidence_mask is None:
        nbar = np.mean(msel[selection_mask]) 
        nmean = np.sum(msel[selection_mask])/np.sum(selection[selection_mask])
        msel=counts_selcorrected(l,b,selection,nside,weight)
        #print(nmean,nbar,nmean/nbar)        
        nbar = nmean        
    else:
        selection_mask = selection_mask&nbar_confidence_mask
        #nbar = np.mean(msel[selection_mask])
        nbar = np.sum(msel[selection_mask])/np.sum(selection[selection_mask])
    msel=counts_selcorrected(l,b,selection,nside,weight)
    if verbose:
        print("nbar",nbar)
    #csel = overdensity_from_counts(msel,nbar,verbose=False) 
    csel = msel/nbar -1
    csel[~selection_mask]=0.

    
    return csel

def process_catalog_splits(l,b,selection,nside,nbar_confidence_mask=None,nreal=1,weight=None,verbose=False):
    selection_mask = selection>0
    if nbar_confidence_mask is not None:
        selection_mask=selection_mask&nbar_confidence_mask
    data=[]
    for i in range(nreal):
        split_reshuffle = np.arange(len(l))
        np.random.shuffle(split_reshuffle)    

        # splits
        m1sel = counts_selcorrected((l[split_reshuffle])[0::2],(b[split_reshuffle])[0::2],selection,nside,weight=weight)
        m2sel = counts_selcorrected((l[split_reshuffle])[1::2],(b[split_reshuffle])[1::2],selection,nside,weight=weight)
    
        c1sel = overdensity_from_counts(m1sel,selection_mask,verbose=verbose) 
        c2sel = overdensity_from_counts(m2sel,selection_mask,verbose=verbose) 
        jksel =(c2sel-c1sel)/2
        
        data.append([c1sel,c2sel,jksel])
    if nreal==1:
        return data[0]
    else:
        return data
    
def process_catalog_and_splits(l,b,selection,nside,nbar_confidence_mask=None,weight=None):
    c =process_catalog(l,b,selection,nside,nbar_confidence_mask=nbar_confidence_mask,weight=weight)
    c1,c2,cjk =process_catalog_splits(l,b,selection,nside,nbar_confidence_mask=nbar_confidence_mask,nreal=1,weight=weight)    
    return c,c1,c2,cjk


def generate_correlated_fields(cls1,cls2,clsx,lmax,nside,pixwin=True):
    if lmax>3*nside-1:
        raise ValueError("lmax > 3nside-1")
    cls_id = np.ones(lmax+1)
    alms_g1 = hp.synalm(cls_id,lmax=lmax)
    alms_g2 = hp.synalm(cls_id,lmax=lmax)
    alms_map1 = hp.almxfl(alms_g1,np.sqrt(cls1[:lmax+1]))
    f1 = clsx[:lmax+1]/np.sqrt(cls1[:lmax+1])
    f2 = np.sqrt(cls2[:lmax+1]-f1**2)
    f1[f1!=f1]=0.
    f2[f2!=f2]=0.
    alms_map2 = hp.almxfl(alms_g1,f1)+hp.almxfl(alms_g2,f2)
    map1,map2 = hp.alm2map([alms_map1,alms_map2],nside=nside,pol=False,lmax=lmax,pixwin=pixwin) 
    return map1,map2


def generate_correlated_field_from_field(map1,cls1,cls2,clsx,lmax,nside,pixwin=True):
    if lmax>3*nside-1:
        raise ValueError("lmax > 3nside-1")
    cls_id = np.ones(lmax+1)
    #alms_g1 = hp.synalm(cls_id,lmax=lmax)
    nside_map1 = hp.npix2nside(len(map1))
    if nside_map1<32:
        # more iteration for better quadrature
        niter=6
    else:
        niter = 3
    alms_map1 = hp.map2alm(map1,lmax=lmax,pol=False,iter=niter)
    if pixwin:
        alms_map1 = hp.almxfl(alms_map1,1/hp.pixwin(nside))
    map1_to_g1 = np.zeros(lmax+1)
    cls1_nanmask = cls1[:lmax+1]!=0
    map1_to_g1[cls1_nanmask] = 1./np.sqrt(cls1[:lmax+1][cls1_nanmask])
    map1_to_g1[map1_to_g1!=map1_to_g1]=0.
    map1_to_g1[np.isinf(map1_to_g1)]=0.

    alms_g1 = hp.almxfl(alms_map1,map1_to_g1)
    alms_g2 = hp.synalm(cls_id,lmax=lmax)    
    f1 = clsx[:lmax+1]/np.sqrt(cls1[:lmax+1])
    f2 = np.sqrt(cls2[:lmax+1]-f1**2)
    f1[f1!=f1]=0.
    f2[f2!=f2]=0.
    alms_map2 = hp.almxfl(alms_g1,f1)+hp.almxfl(alms_g2,f2)
    map2 = hp.alm2map(alms_map2,nside=nside,pol=False,lmax=lmax,pixwin=pixwin) 
    return map2


def prepare_sysmap(mp, mask):
    # from David's notebook for consistency
    # there, mask is the QSO mask multiplied by the  selection function
    mean = np.sum(mp*mask)/np.sum(mask)
    mp_out = mp-mean
    mp_out[mask <= 0] = 0
    return mp_out
