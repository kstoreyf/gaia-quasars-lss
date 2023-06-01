import numpy as np
import healpy as hp
import timeit
import sys
import pickle as pkl
import os
from astropy.io import fits 
from astropy.coordinates import SkyCoord
import argparse 
sys.path.append('../src/')
sys.path.append('../code')

from lensing_qso_cross_utils import *
from lensing_qso_cross_utils_qml import *
import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
from xqml.libcov import compute_ds_dcb

import pylab as pl 

def get_binning_custom():
    lmins = np.arange(2,40,2)
    lmins = np.append(lmins,np.arange(40,60,5))
    lmins = np.append(lmins,np.arange(60,767,30))

    lmaxs = np.arange(4,42,2)
    lmaxs = np.append(lmaxs,np.arange(45,65,5))
    lmaxs = np.append(lmaxs,np.arange(90,767,30))
    lmaxs = np.append(lmaxs,767)
    return lmins,lmaxs


def get_binning_desi():
    lmins = np.arange(2,20,2)
    lmins = np.append(lmins,np.arange(20,100,5))
    lmins = np.append(lmins,np.arange(100,767,20))

    lmaxs = np.arange(4,22,2)
    lmaxs = np.append(lmaxs,np.arange(25,105,5))
    lmaxs = np.append(lmaxs,np.arange(120,767,20))
    lmaxs = np.append(lmaxs,767)
    return lmins,lmaxs

rot = hp.Rotator(coord=['C','G'])

patch = "namaster"

def main(args):
    masks_list = args.mask.split(',')
    masks = [int(m[-2:]) for m in masks_list]
    mask_type = masks_list[0][:-2]
    
    np.random.seed(123456)

    nsims = args.nsim
    nside = args.nside_lr
    nside_hr = args.nside_hr
    dell = 2 #binning
    lmin = 2 # doesn't matter as code forces lmin=2 and cls assum l_0=l_1=0.
    lmax = 3 * nside - 1
    lmax_hr = 3 * nside_hr -1
    nreal_galsplits = args.n_galsplit

    do_g_syst = False
    if nside<32:
        niter_sht=6
    else:
        niter_sht=3

    # CMB lensing kinematic dipole
    aberration_lbv_ffp10 = (264. * (np.pi / 180), 48.26 * (np.pi / 180), 0.001234)
    l_ffp10, b_ffp10, v_ffp10 = aberration_lbv_ffp10
    vlm_ffp10 = np.array([0., np.cos(b_ffp10), - np.exp(-1j * l_ffp10) * np.sin(b_ffp10) / np.sqrt(2.)])
    vlm_ffp10 *= (-v_ffp10 * np.sqrt(4 * np.pi / 3))
    v_ffp10  = hp.alm2map(vlm_ffp10,nside,pixwin=True,lmax=1)
    v_ffp10_hr  = hp.alm2map(vlm_ffp10,nside_hr,pixwin=True,lmax=1)
    if args.nodipole_mocks:
        v_ffp10 *= 0.
        v_ffp10_hr *= 0.

    # systematics maps
    if args.systematics:
        do_g_syst=False
        syst=[]
        if 'dust' in args.systematics:
            syst.append(rot.rotate_map_pixel(np.load('%s/map_dust_NSIDE64.npy'%args.catalog_path)))
        if 'm10' in args.systematics:    
            m10 = rot.rotate_map_pixel(np.load('%s/map_m10_NSIDE64.npy'%args.catalog_path))
        if 'mcs' in args.systematics:    
            mcs = rot.rotate_map_pixel(np.load('%s/map_mcs_NSIDE64.npy'%args.catalog_path))
            mcs[mcs<0] = 0.
        if 'star' in args.systematics:   
            syst.append(rot.rotate_map_pixel(np.load('%s/map_stars_NSIDE64.npy'%args.catalog_path)))
        if 'star_wise' in args.systematics:   
            syst.append(rot.rotate_map_pixel(hp.read_map('%s/allwise_total_rot_1024.fits'%args.catalog_path)))            
    
        if not len(syst):
            raise ValueError("Systematics list not including valuable templates: %s"%args.systematics) 
        syst_lr = [hp.ud_grade(s,nside_out=nside) for s in syst]
        syst_hr = [hp.ud_grade(s,nside_out=nside_hr) for s in syst]

    else:
        syst=[]
        do_g_syst=False

    g_maps = args.catalog.split(',')
    noise_models=args.noise_type.split(',')

    mask_file = '%s/mask_%s_GalPlane_ns16_R2.00_ring_fsky%s.fits'%(args.mask_path,mask_type,"%d")

    lens_mask_hr = hp.read_map('%s/planck_lensmask.fits'%args.mask_path)
    source_mask_hr = lens_mask_hr*get_magellanic_cloud_mask(nside_hr,r_mclouds=[4,4])
    mask_file_hr = '%s/planck_galmask%s.fits'%(args.mask_path,'%d')
    lens_mask = hp.ud_grade(lens_mask_hr,nside_out=nside) >0.4
    source_mask = lens_mask * get_magellanic_cloud_mask(nside,r_mclouds=[4,4])
    galmask80 = hp.read_map('%s/planck_galmask80.fits'%args.mask_path)
    galmask80_lr= hp.ud_grade(galmask80.astype(np.float32),nside_out=nside) >0.4

    # selection function and binary mask
    g_map = g_maps[0]
    mask_c = hp.ud_grade(hp.read_map('%s/selection_function_NSIDE64_%s.fits'%(args.catalog_path,g_map)),nside_out=256)
    selfunc_hr = rot.rotate_map_pixel(mask_c)
    selfunc = hp.ud_grade(selfunc_hr,nside_out=nside)
    binmask_hr = (selfunc_hr>0) & galmask80
    binmask = (selfunc>0) & galmask80_lr
    
    
    # Randomized catalog
    drnd=fits.open('%s/random_%s_10x.fits'%(args.catalog_path,g_map))
    scrnd = SkyCoord(ra=drnd[1].data["ra"], dec=drnd[1].data["dec"], unit='deg', frame='icrs', equinox='J2000.0')
    gsrnd = scrnd.transform_to(frame='galactic')
    nqso_rnd = len(gsrnd.l.value)
    split_reshuffle_rnd = np.arange(nqso_rnd)

    np.random.shuffle(split_reshuffle_rnd)

    if args.binning == 'custom':
        lmins,lmaxs = get_binning_custom()
        bins_nmt= nmt.NmtBin.from_edges(lmins, lmaxs, is_Dell=False)
        bins = xqml.Bins(lmins[lmaxs<lmax],lmaxs[lmaxs<lmax])                
    elif args.binning == 'desi':
        lmins,lmax = get_binning_desi()
        bins_nmt= nmt.NmtBin.from_edges(lmins, lmaxs, is_Dell=False)
        bins = xqml.Bins(lmins[lmaxs<lmax],lmaxs[lmaxs<lmax])        
    else:
        dell = int(args.binning)
        lmins = np.arange(0,lmax,dell)
        lmaxs = np.arange(0+dell,lmax+dell,dell)
        lmins[0]=2
        lmaxs[-1]=lmax
        bins = xqml.Bins(lmins,lmaxs)
        bins = xqml.Bins.fromdeltal( lmin, lmax, dell)
        bins_nmt = get_custom_binning(dell,weighting='flat')
    
    lb = bins.lbin
    lb_nmt = bins_nmt.get_effective_ells()

    
    pixwin = False
    if pixwin:
        fpixw = extrapolpixwin(nside, lmax, pixwin=pixwin)
        bl = np.ones(lmax+1)
    else:
        bl = hp.pixwin(nside,pol=False)[:lmax+1]

    spec=['TT']
    stokes, spec, istokes, ispecs = getstokes( spec=spec)
    nspec = len(spec)
    nstoke = len(stokes)


    ##############################
    #Prepare lensing data
    print("Read k maps")
    #klr=prepare_k_data(nside,'/Users/gfabbian/Work/PR4_variations/PR42018like_klm_dat_p.fits',pixwin=None,overwrite=True)
    klr=prepare_k_data(nside,'%s/%s_klm_dat_p.fits'%(args.lensing_path,args.lensing_map,),pixwin=1,overwrite=True,remove_dipole=args.nodipole_mocks)  # extrapolated pixelwindow
    k_nmt=prepare_k_data(nside_hr,'%s/%s_klm_dat_p.fits'%(args.lensing_path,args.lensing_map,),pixwin=1,overwrite=True,remove_dipole=args.nodipole_mocks)  # extrapolated pixelwindow

    if args.nodipole_mocks:
        dipole_suffix = '_nodip'
    else:
        dipole_suffix = ''
    
    print("Read k mc nside",nside,nside_hr)
    with open("%s/sims/%s_k_sims_nside%d%s.pkl"%(args.lensing_path,args.lensing_map,nside,dipole_suffix),"rb") as f:
        kmc = pkl.load(f)
    with open("%s/sims/%s_k_sims_nside%d%s.pkl"%(args.lensing_path,args.lensing_map,nside_hr,dipole_suffix),"rb") as f:
        kmc_hr = pkl.load(f)[:nsims]

    k_cov = np.var(kmc,axis=0)
    k_covmat = np.cov(kmc,rowvar=False)
    k_corrmat = np.corrcoef(kmc,rowvar=False)
    kmap_dic_key ='k'

    # read ffp10 signal only sims
    with open("%s/sims/ffp10_k_sims_nside%d%s.pkl"%(args.lensing_path,nside,dipole_suffix),"rb") as f:
        kmc_signal = pkl.load(f) 

    with open("%s/sims/ffp10_k_sims_nside%d%s.pkl"%(args.lensing_path,nside_hr,dipole_suffix),"rb") as f:
        kmc_signal_hr = pkl.load(f)[:nsims]

    N_k_covmat = np.cov(kmc-kmc_signal,rowvar=False)

    #print(np.sum(k_cov),np.sum(k_covmat.diagonal()))
    #assert np.sum(k_cov) == np.sum(k_covmat.diagonal())

    ##############################
    #input model
    with open(args.cl_theory,"rb") as f:
        clth = pkl.load(f)
    
    lth = np.arange(len(clth['PxP']))
    clkk = clth['PxP']*(lth*(lth+1)/2)**2
    clgg = clth['W1xW1']
    clkg = clth['PxW1']*lth*(lth+1)/2

    Dldd_ffp10=np.append([0],np.loadtxt('%s/FFP10_wdipole_lenspotentialCls.dat'%args.lensing_path,unpack=True,usecols=5))[:len(clth['PxP'])]
    clkk = Dldd_ffp10*np.pi/2
    #nlkk=np.loadtxt('/Users/gfabbian/Work/PR4_variations/PR42018like_nlkk_p.dat',unpack=True)
    #nlkk=np.loadtxt('/Users/gfabbian/Work/PR4_variations/PR4_nlkk_p.dat',unpack=True)
    p2d=np.sqrt(lth*(lth+1))
    d2p=1/p2d
    d2p[0:2]=0.0
    d2k=p2d/2
    p2k=(p2d**2/2)[:lmax+1]

    ############################

    k_real=np.zeros(hp.nside2npix(nside))
    g_real=np.zeros(hp.nside2npix(nside))

    for remove_dipole_monopole in [args.marginalize_dipole]:#,False]:
        #print("Maginalize dipole",remove_dipole_monopole)
        for j,g_map_name in enumerate(g_maps):

            # Prepare raw GAIA data
            print("Read catalog %s and convert coordinate"%g_map_name)
            d=fits.open('%s/catalog_%s.fits'%(args.catalog_path,g_map_name))
            nqso = len(d[1].data['redshift_spz'])
            print ("N_QSO in catalog",nqso)

            # redo shit for catalog
            sc = SkyCoord(ra=d[1].data["ra"], dec=d[1].data["dec"], unit='deg', frame='icrs', equinox='J2000.0')
            gs = sc.transform_to(frame='galactic')
            l = gs.l.value 
            b = gs.b.value # d[1].data['b'])

            split_reshuffle = np.arange(nqso)

            np.random.shuffle(split_reshuffle)

            for mask_name in masks:          
                np.random.seed(123456)
                if 'planck' in mask_file:
                    # added safety check
                    gmask_lr = hp.read_map(mask_file%mask_name).astype(bool) & (selfunc_hr>0.5)
                    gmask_hr = hp.read_map(mask_file_hr%mask_name) & (selfunc>0.5)
                else:
                    gmask_lr = selfunc > mask_name/100.
                    gmask_hr = selfunc_hr > mask_name/100.
                print("fsky mask %s"%(args.mask),np.mean(gmask_lr),np.mean(gmask_hr))
                mask_hr = gmask_hr * source_mask_hr 
                mask = gmask_lr & source_mask.astype(bool)                
                apomask_hr = nmt.mask_apodization(mask_hr,1., apotype="C2")
                npix = int(np.sum(mask))
                print("fsky total mask",np.mean(mask),np.mean(mask_hr))
                
                data_in_mask = process_catalog_and_splits(l,b,selfunc_hr,nside=nside_hr,nbar_confidence_mask=mask_hr.astype(bool))

                g_map_sel = hp.ud_grade(data_in_mask[0],nside_out=nside)
                #g_map1_sel = hp.ud_grade(data_in_mask[1],nside_out=nside)
                #g_map2_sel = hp.ud_grade(data_in_mask[2],nside_out=nside)
                #gjk_map_sel = hp.ud_grade(data_in_mask[3],nside_out=nside)
                #gjk_map_sel =(g_map2_sel-g_map1_sel)/2

                msel_hr = counts_selcorrected(l,b,selfunc_hr*mask_hr.astype(np.float32),nside=nside_hr)
                msel_lr = counts_selcorrected(l,b,selfunc*mask.astype(np.float32),nside=nside)
                msel = hp.ud_grade(msel_hr,nside_out=nside,power=-2)

                nbar_sel = np.mean(msel[mask])
                nbar_hr = np.mean(msel_hr[mask_hr>0])
                nbar_lr = np.mean(msel_lr[mask>0])
                assert np.sum(msel_hr) == np.sum(msel)
                print("nbar hr",nbar_hr,"nbar lr (HR degraded)",nbar_sel,"nbar lr",nbar_lr)
                print("Shot noise lr",hp.nside2pixarea(nside)/nbar_sel)
                print("Shot noise hr",hp.nside2pixarea(nside_hr)/nbar_hr)

                gmap_dic_key = g_map_name

                fsky = np.mean(mask)
                npix = np.sum(mask)

                #out_fname = '/Users/gfabbian/Work/quasar_gaia/output/cls_kXQSO_%s_planck%d_remove_dipole_%s_knlgcovselfunc_dl20.pkl'%(g_map_name.replace('.','p'),mask_name,str(remove_dipole_monopole))
                #out_fname = '/Users/gfabbian/Work/quasar_gaia/output/PR4_%s_60_nogpixwin_gdipole_dl%d_nmt_alldip10k_nokdip.pkl'%(str(remove_dipole_monopole),dell)
                out_fname = args.out_file

                if False:#os.path.exists(out_fname):
                    with open(out_fname,"rb") as f:
                        data_pkl=pkl.load(f)
                    data_pkl['lb'] = lb
                    data_pkl['lbounds'] = [lmin,lmax]
                    data_pkl['dell'] = dell                    
                else:
                    data_pkl={}
                    data_pkl={k:{'data':{},'sims':{}} for k in noise_models}
                    data_pkl['data_format'] = ['cl_kk cl_gg cl_kg cl_kxk cl_gxg']
                    data_pkl['lb'] = lb
                    data_pkl['lbounds'] = [lmin,lmax]
                    data_pkl['dell'] = dell
                    data_pkl['monopole_dipole']={}

                    keys_nmt = [('k','k'),('k','g_sel'),('g_sel','g_sel'),('g1_sel','g1_sel'),('g2_sel','g2_sel'),('g1_sel','g2_sel'),('gjk_sel','gjk_sel'),('k','gjk_sel'),('k','k_in'),('k_in','k_in')]
                    #data_nmt = {'data':{},'sims':{}}
                    data_nmt={}
                    for k in noise_models:
                        data_nmt[k]={'data':{},'sims':{}}
                        for key_nmt in keys_nmt:
                            data_nmt[k]['sims'][key_nmt]=[]
                ###########################
                # memory check
                if args.verbose:
                    print("%d mask: fsky=%.2g %% (npix=%d)" % (mask_name,100*fsky,npix))
                    toGB = 1024. * 1024. * 1024.
                    emem = 8.*(npix*nstoke*npix*nstoke) * ( len(lth)*2 ) / toGB
                    print("mem=%.2g Gb" % emem)
                ##############################
                if remove_dipole_monopole:
                    # TBD
                    vecs=np.array(hp.pix2vec(nside,np.arange(hp.nside2npix(nside))[mask]))
                    dipole_cov = np.dot(vecs.T,vecs)*3/4/np.pi #+0.02*np.random.normal(size=(npix,npix))
                    #dipole_cov = np.linalg.inv(dipole_cov)

                    # repeat for selection function
                    g_real[:] = hp.UNSEEN
                    g_real[mask] = g_map_sel[mask]
                    g_map_buf,mono_sel,dipole_sel=hp.pixelfunc.remove_dipole(g_real,fitval=True)
                    data_pkl['monopole_dipole']['g_sel']=(mono_sel,dipole_sel)

                #shortens maps and redefines them as 2-d array with 1 row
                g_map_sel_run=np.array([g_map_sel[mask]])            
                k_map_run=np.array([klr[mask]])

                ### define Gaia covariance
                ### compute covariance
                ### define masked array
                if remove_dipole_monopole:
                    g_real[:]=hp.UNSEEN
                    g_real[mask] = g_map_sel[mask]
                    gmap_buf,mono,dipole=hp.pixelfunc.remove_dipole(g_real,fitval=True)
                    data_pkl['monopole_dipole'][gmap_dic_key]=(mono,dipole)

                S_k = compute_ds_dcb(bins,nside,np.where(mask>0)[0],bl, np.array([clkk]),lmax,spec,pixwin=pixwin,Sonly=True)
                S_g = compute_ds_dcb(bins,nside,np.where(mask>0)[0],bl, np.array([clgg]),lmax,spec,pixwin=pixwin,Sonly=True)
                #N_k = compute_ds_dcb(bins,nside,np.where(mask>0)[0],bl, np.array([nlkk]),lmax,spec,pixwin=pixwin,Sonly=True)

                for noise in noise_models:
                    # ############## Generate White Noise ###############
                    if 'diagk' in noise:
                        #NoiseVar = np.diag(k_covmat.diagonal()[mask]) -S_k # uses full covariance from simulations with signal and noise
                        NoiseVar = np.diag(N_k_covmat.diagonal()[mask])
                    else:
                        NoiseVar = N_k_covmat[np.outer(mask,mask)]
                        dim=int(np.sqrt(NoiseVar.shape[0]))
                        NoiseVar = NoiseVar.reshape(dim,dim)
                    if 'diagg' in noise:
                        #NoiseVar_g = np.diag(hp.nside2pixarea(nside)*np.ones(npix)/nbar) # wrong, no pixarea factor needed
                        # homogenous weighting                   
                        NoiseVar_g = np.diag(np.ones(npix)/nbar_sel) # homogenous weighting after sel. func correction  
                        weight_hr = np.ones(hp.nside2npix(nside_hr))
                    elif 'corrg' in noise:
                        ## accounts for dishomogeneous nbar
                        #counts_rnd_g = make_counts(nside,gsrnd.l.value,gsrnd.b.value)[mask]*nqso/nqso_rnd
                        counts_rnd_g_sel = make_counts(nside,gsrnd.l.value,gsrnd.b.value)[mask]*nqso/nqso_rnd/selfunc[mask]
                        NoiseVar_g = np.diag(1/counts_rnd_g_sel)                                       
                        weight_hr = np.array(selfunc_hr)
                    else:
                        raise ValueError("Unknown noise model for galaxies or lensing %s"%noise)

                    # ############## Initialise xqml class ###############
                    print(g_map_name,noise,'remove dipole',remove_dipole_monopole)
                    if remove_dipole_monopole:
                        #g_dipole_rescale = 1000*dipole_cov
                        #NoiseVar_g += (g_dipole_rescale*dipole_cov)
                        k_dipole_rescale=10000*dipole_cov
                        g_dipole_rescale = 10000*dipole_cov
                        #k_dipole_rescale=1000*np.outer(v_ffp10[mask],v_ffp10[mask])
                        #g_dipole_rescale = 50*np.outer(v_ffp10[mask],v_ffp10[mask])
                    else:
                        k_dipole_rescale = 0.
                        g_dipole_rescale = 0.
                    if do_g_syst:
                        for t in syst_lr:
                           local_t = prepare_sysmap(t, mask)
                           g_dipole_rescale+= 1000*np.outer(local_t[mask],local_t[mask])
                        nmt_syst = [[prepare_sysmap(t,((mask_hr>0) & (t>0)))] for t in syst_hr]
                    else:
                        nmt_syst = None

                    low_noise_rescaling=100
                    #low_noise_rescaling=1000

                    k_est = xqml.xQML(mask, bins, np.array([clkk]), NA=NoiseVar+k_dipole_rescale, lmax=lmax, bell=bl, spec=spec,S=S_k,verbose=False,pixwin=pixwin)
                    kffp10_est = xqml.xQML(mask, bins, np.array([clkk]), NA=NoiseVar/low_noise_rescaling +k_dipole_rescale, lmax=lmax, bell=bl, spec=spec,S=S_k,Pl=k_est.Pl,verbose=False,pixwin=pixwin)
                    kXkffp10_est = xqml.xQML(mask, bins, np.array([clkk]), NA=NoiseVar +k_dipole_rescale, NB=NoiseVar/low_noise_rescaling +k_dipole_rescale, lmax=lmax, bell=bl, spec=spec,S=S_k,Pl=k_est.Pl,verbose=False,pixwin=pixwin)

                    gg_est = xqml.xQML(mask, bins, np.array([clgg]), NA=NoiseVar_g+g_dipole_rescale, lmax=lmax, bell=bl, spec=spec,verbose=False,S=S_g,Pl=k_est.Pl,pixwin=pixwin)

                    SXcov = k_est.S
                    kg_est = xqml.xQML(mask, bins, np.array([clkg]), NA=NoiseVar, NB=NoiseVar_g+g_dipole_rescale -SXcov + S_g, lmax=lmax, bell=bl, spec=spec,Pl=k_est.Pl,
                        S=SXcov,verbose=False,pixwin=pixwin) #np.sqrt(hp.pixwin(nside,pol=False)[:lmax+1])


                    gXg_est = xqml.xQML(mask, bins, np.array([clgg]), NA=NoiseVar_g*2+g_dipole_rescale, lmax=lmax, bell=bl, spec=spec,S=S_g,Pl=k_est.Pl,
                        verbose=False,pixwin=pixwin)

                    # compute k data
                    cl_kk = k_est.get_spectra(k_map_run)[0]
                    auto_kk = k_est.get_spectra(k_map_run,k_map_run)[0]

                    # compute g data with selection function
                    cl_gg_sel = gg_est.get_spectra(g_map_sel_run)[0]
                    auto_gg_sel = gg_est.get_spectra(g_map_sel_run,g_map_sel_run)[0]

                    # compute kg cross-correlation
                    cl_kg_sel = kg_est.get_spectra(k_map_run,g_map_sel_run)[0]

                    # loop over realizations of splits and computing the mean for g debiased spectra
                    cl_gjk_sel=np.zeros_like(cl_kk)
                    auto_gjk_sel=np.zeros_like(cl_kk)
                    cl_kgjk_sel=np.zeros_like(cl_kk)
                    cl_gXg_sel=np.zeros_like(cl_kk)
                    cl_g1_sel=np.zeros_like(cl_kk)
                    cl_g2_sel=np.zeros_like(cl_kk)

                    clkg_nmt,clgg_nmt,clkk_nmt,clkg1_nmt,clg1g1_nmt,clkg2_nmt,clg2g2_nmt,clg1g2_nmt,clkgjk_nmt,clgjk_nmt,w = compute_master_crosscorr_mask(k_nmt,data_in_mask[0],data_in_mask[1],
                                                                                data_in_mask[2],data_in_mask[3],apodized_mask=apomask_hr*weight_hr,binning=bins_nmt,lmax=lmax_hr,gsyst=nmt_syst,return_mode_coupling=True,w=None)
                    #wins = w.get_bandpower_windows()
                    #print(wins.squeeze().shape)
                    #pl.plot(wins.squeeze()[0])
                    #pl.plot(wins.squeeze()[1])
                    #pl.plot(wins.squeeze()[10])
                    #pl.show()
                    #dada

                    clkgjk_nmt=0.
                    clgjk_nmt=0.
                    clg1g2_nmt=0.
                    clg1g1_nmt=0.
                    clg2g2_nmt=0.

                    print("Computing noise debiased g spectra on %d splits",nreal_galsplits)
                    for n in range(nreal_galsplits):
                        if args.verbose:
                            progress_bar(n, nreal_galsplits)
                        data_in_mask = process_catalog_and_splits(l,b,selfunc_hr,nside=nside_hr,nbar_confidence_mask=mask_hr.astype(bool))

                        g_map1_sel = hp.ud_grade(data_in_mask[1],nside_out=nside)
                        g_map2_sel = hp.ud_grade(data_in_mask[2],nside_out=nside)
                        gjk_map_sel = hp.ud_grade(data_in_mask[3],nside_out=nside)

                        g_map1_sel_run=np.array([g_map1_sel[mask]])
                        g_map2_sel_run=np.array([g_map2_sel[mask]])
                        gjk_map_sel_run=np.array([gjk_map_sel[mask]])

                        cl_gjk_sel += (gg_est.get_spectra(gjk_map_sel_run)[0]/nreal_galsplits)
                        auto_gjk_sel += (gg_est.get_spectra(gjk_map_sel_run,gjk_map_sel_run)[0]/nreal_galsplits)

                        cl_kgjk_sel += (kg_est.get_spectra(k_map_run,gjk_map_sel_run)[0]/nreal_galsplits)
                        cl_gXg_sel += (gXg_est.get_spectra(g_map1_sel_run,g_map2_sel_run)[0]/nreal_galsplits)
                        cl_g1_sel += (gXg_est.get_spectra(g_map1_sel_run,g_map1_sel_run)[0]/nreal_galsplits)
                        cl_g2_sel += (gXg_est.get_spectra(g_map2_sel_run,g_map2_sel_run)[0]/nreal_galsplits)

                        cls_data_nmt = compute_master_crosscorr_mask(k_nmt,data_in_mask[0],data_in_mask[1],data_in_mask[2],data_in_mask[3],apodized_mask=apomask_hr*weight_hr,binning=bins_nmt,lmax=lmax_hr,gsyst=nmt_syst,w=w)

                        clg1g1_nmt+=cls_data_nmt[4]/nreal_galsplits
                        clg2g2_nmt+=cls_data_nmt[6]/nreal_galsplits
                        clg1g2_nmt+=cls_data_nmt[-3]/nreal_galsplits
                        clkgjk_nmt+=cls_data_nmt[-2]/nreal_galsplits
                        clgjk_nmt+= cls_data_nmt[-1]/nreal_galsplits
                    print("done")

                    # ############## Construct MC ###############
                    if nsims>0: 
                        allclk = []
                        allclk_auto = []
                        allclkXks = []     # noisy X signbal
                        allclks_auto = []   #signal
                        allclg = []
                        allclg_auto = []
                        allclgjk = []
                        allclgjk_auto = []
                        allclkg = []
                        allclkgjk = []
                        allclgXg = []
                        allclgth = []

                        allclg_sel = []
                        allclg_sel_auto = []
                        allclgjk_sel = []
                        allclgjk_sel_auto = []
                        allclkg_sel = []
                        allclkgjk_sel = []
                        allclgXg_sel = []
                        allclgth_sel = []
                        allclg1_sel_auto = []
                        allclg2_sel_auto = []

                        cl_theory_mc =np.zeros((3,3*nside))

                        t = []
                        km = np.zeros((1,npix))
                        km_signal = np.zeros((1,npix))

                        print("Processing %d simulations",nsims)
                        for n in range(nsims):
                            if args.verbose:
                                progress_bar(n, nsims)

                            # random realization of kappa and g adding gaussian N0
                            #k,g= generate_correlated_fields(clkk,clgg,clkg,3*nside-1,nside,pixwin=True)
                            #km[0,:]=k[mask] + hp.synfast(nlkk[:lmax+1],nside,pol=False,pixwin=pixwin)[mask]

                            km[0,:] = kmc[n][mask] # from ffp10 sims
                            km_signal[0,:] = kmc_signal[n][mask] # from ffp10 sims
                            g = generate_correlated_field_from_field(kmc_signal[n]-v_ffp10,clkk,clgg,clkg,lmax,nside,pixwin=True) #correlated galaxy overdensity from ffp10

                            km_hr = kmc_hr[n]
                            #g_hr = hp.ud_grade(g,nside_out=256)
                            g_hr = generate_correlated_field_from_field(kmc_signal_hr[n]-v_ffp10_hr,clkk,clgg,clkg,lmax_hr,nside_hr,pixwin=True) #correlated galaxy overdensity from ffp10
                            fl_hr_to_lr =1/hp.pixwin(nside_hr,lmax=lmax_hr)#np.ones(lmax_hr+1)
                            fl_hr_to_lr[lmax+1:]=0.
                            fl_hr_to_lr[:lmax+1]=hp.pixwin(nside,lmax=lmax)
                            alm_ghr_lr = hp.almxfl(hp.map2alm(g_hr,lmax=lmax_hr,pol=False),fl_hr_to_lr,)
                            # reassigns to 
                            g = hp.alm2map(alm_ghr_lr,nside,lmax=lmax_hr,pixwin=False)

                            alms_mc_maps = hp.map2alm([kmc_signal[n],g],pol=False,iter=niter_sht)
                            #alms_mc_maps_hr = hp.map2alm([kmc_signal_hr[n],g],pol=False,iter=niter_sht)

                            cl_theory_mc += np.array([hp.alm2cl(alms_mc_maps[0]),hp.alm2cl(alms_mc_maps[1]),hp.alm2cl(alms_mc_maps[0],alms_mc_maps[1])])/nsims

                            gm=np.array(g)[mask]

                            np.random.shuffle(split_reshuffle_rnd)

                            #km_signal[0,:]+= np.random.normal(size=npix)*np.sqrt(NoiseVar.diagonal()/100)

                            lmc = (gsrnd.l.value[split_reshuffle_rnd])
                            bmc = (gsrnd.b.value[split_reshuffle_rnd])

                            goodpix = mask>0

                            mc_in_mask = process_catalog_and_splits(lmc[:nqso],bmc[:nqso],np.ones(hp.nside2npix(256)),nside=nside_hr,nbar_confidence_mask=mask_hr.astype(bool))

                            crnd_mc = hp.ud_grade(mc_in_mask[0],nside_out=nside)[mask] + gm
                            crnd1_mc = hp.ud_grade(mc_in_mask[1],nside_out=nside)[mask] + gm
                            crnd2_mc = hp.ud_grade(mc_in_mask[2],nside_out=nside)[mask] + gm
                            jkrnd_mc = hp.ud_grade(mc_in_mask[3],nside_out=nside)[mask] + gm

                            mcsel_in_mask = process_catalog_and_splits(lmc[:nqso],bmc[:nqso],selfunc_hr,nside=256,nbar_confidence_mask=mask_hr.astype(bool))
                            cselrnd_mc = hp.ud_grade(mcsel_in_mask[0],nside_out=nside)[mask] + gm
                            cselrnd1_mc = hp.ud_grade(mcsel_in_mask[1],nside_out=nside)[mask] + gm
                            cselrnd2_mc = hp.ud_grade(mcsel_in_mask[2],nside_out=nside)[mask] + gm
                            jkselrnd_mc = hp.ud_grade(mcsel_in_mask[3],nside_out=nside)[mask] + gm

                            cls_sims_nmt = compute_master_crosscorr_mask(km_hr,mcsel_in_mask[0]+g_hr,mcsel_in_mask[1]+g_hr,mcsel_in_mask[2]+g_hr,mcsel_in_mask[3]+g_hr,apodized_mask=apomask_hr,
                                                                         binning=bins_nmt,lmax=lmax_hr,gsyst=nmt_syst,w=w)

                            data_nmt[noise]['sims'][(kmap_dic_key,kmap_dic_key)].append(cls_sims_nmt[2])
                            data_nmt[noise]['sims'][('g_sel','g_sel')].append(cls_sims_nmt[1])

                            data_nmt[noise]['sims'][(kmap_dic_key,'g_sel')].append(cls_sims_nmt[0])
                            data_nmt[noise]['sims'][('g1_sel','g1_sel')].append(cls_sims_nmt[4])
                            data_nmt[noise]['sims'][('g2_sel','g2_sel')].append(cls_sims_nmt[6])
                            data_nmt[noise]['sims'][('g1_sel','g2_sel')].append(cls_sims_nmt[7])

                            data_nmt[noise]['sims'][(kmap_dic_key,'gjk_sel')].append(cls_sims_nmt[8])
                            data_nmt[noise]['sims'][('gjk_sel','gjk_sel')].append(cls_sims_nmt[9])

                            # cross-correlation  for MC correction 
                            cls_sims_nmt = compute_master_crosscorr_mask(km_hr,kmc_signal_hr[n],mcsel_in_mask[1]+g_hr,mcsel_in_mask[2]+g_hr,mcsel_in_mask[3]+g_hr,apodized_mask=apomask_hr,binning=bins_nmt,lmax=lmax_hr)

                            data_nmt[noise]['sims'][(kmap_dic_key,'k_in')].append(cls_sims_nmt[0])
                            data_nmt[noise]['sims'][('k_in','k_in')].append(cls_sims_nmt[1])

                            # compute sims k
                            allclk.append(k_est.get_spectra(km)[0])
                            allclk_auto.append(k_est.get_spectra(km,km)[0])
                            allclkXks.append(kXkffp10_est.get_spectra(km,km_signal)[0])
                            allclks_auto.append(kffp10_est.get_spectra(km_signal,km_signal)[0])
                            # with selection function

                            allclg_sel.append(gg_est.get_spectra(cselrnd_mc)[0])
                            allclg_sel_auto.append(gg_est.get_spectra(cselrnd_mc,cselrnd_mc)[0])
                            allclkg_sel.append(kg_est.get_spectra(km,cselrnd_mc)[0])
                            allclgXg_sel.append(gXg_est.get_spectra(cselrnd1_mc,cselrnd2_mc)[0])
                            allclg1_sel_auto.append(gXg_est.get_spectra(cselrnd1_mc,cselrnd1_mc)[0])
                            allclg2_sel_auto.append(gXg_est.get_spectra(cselrnd2_mc,cselrnd2_mc)[0])

                            #JK
                            allclgjk_sel.append(gg_est.get_spectra(jkselrnd_mc)[0])
                            allclgjk_sel_auto.append(gg_est.get_spectra(jkselrnd_mc,jkselrnd_mc)[0])
                            allclkgjk_sel.append(kg_est.get_spectra(km,jkselrnd_mc)[0])
                            #s1 = timeit.default_timer()


                        # regular
                        data_pkl[noise]['sims'][('k','k')]=allclk_auto
                        data_pkl[noise]['sims'][('k_debias','k_debias')]=allclk
                        data_pkl[noise]['sims'][('k','k_in')]=allclkXks
                        data_pkl[noise]['sims'][('k_in','k_in')]=allclks_auto

                        # selection function 
                        data_pkl[noise]['sims'][('g_sel','g_sel')]=allclg_sel_auto
                        data_pkl[noise]['sims'][('g_sel_debias','g_sel_debias')]=allclg_sel
                        data_pkl[noise]['sims'][('k','g_sel')]=allclkg_sel                    
                        data_pkl[noise]['sims'][('g1_sel','g2_sel')]=allclgXg_sel
                        data_pkl[noise]['sims'][('g1_sel','g1_sel')]=allclg1_sel_auto
                        data_pkl[noise]['sims'][('g2_sel','g2_sel')]=allclg2_sel_auto

                        data_pkl[noise]['sims'][('gjk_sel','gjk_sel')]=allclgjk_sel_auto
                        data_pkl[noise]['sims'][('gjk_sel_debias','gjk_sel_debias')]=allclgjk_sel
                        data_pkl[noise]['sims'][('k','gjk_sel')]=allclkgjk_sel   

                    np.save('cl_theory_%s'%args.out_file,cl_theory_mc)
                    data_pkl[noise]['data'][(kmap_dic_key,kmap_dic_key)]=auto_kk
                    data_pkl[noise]['data'][(kmap_dic_key+'_debias',kmap_dic_key+'_debias')]=cl_kk

                    # sel 
                    data_pkl[noise]['data'][('g_sel','g_sel')]=auto_gg_sel
                    data_pkl[noise]['data'][('g_sel_debias','g_sel_debias')]=cl_gg_sel

                    data_pkl[noise]['data'][(kmap_dic_key,'g_sel')]=cl_kg_sel
                    data_pkl[noise]['data'][('g1_sel','g2_sel')]=cl_gXg_sel   
                    data_pkl[noise]['data'][('g1_sel','g1_sel')]=cl_g1_sel   
                    data_pkl[noise]['data'][('g2_sel','g2_sel')]=cl_g2_sel                

                    data_pkl[noise]['data'][(kmap_dic_key,'gjk_sel')]=cl_kgjk_sel
                    data_pkl[noise]['data'][('gjk_sel','gjk_sel')]=auto_gjk_sel
                    data_pkl[noise]['data'][('gjk_sel_debias','gjk_sel_debias')]=cl_gjk_sel


                    data_pkl[noise]['data'][(kmap_dic_key,kmap_dic_key)]=auto_kk
                    data_pkl[noise]['data'][(kmap_dic_key+'_debias',kmap_dic_key+'_debias')]=cl_kk

                    # sel nmt
                    data_nmt[noise]['data'][(kmap_dic_key,kmap_dic_key)]=clkk_nmt
                    data_nmt[noise]['data'][('g_sel','g_sel')]=clgg_nmt

                    data_nmt[noise]['data'][(kmap_dic_key,'g_sel')]=clkg_nmt
                    data_nmt[noise]['data'][('g1_sel','g1_sel')]=clg1g1_nmt
                    data_nmt[noise]['data'][('g2_sel','g2_sel')]=clg2g2_nmt
                    data_nmt[noise]['data'][('g1_sel','g2_sel')]=clg1g2_nmt

                    data_nmt[noise]['data'][(kmap_dic_key,'gjk_sel')]=clkgjk_nmt
                    data_nmt[noise]['data'][('gjk_sel','gjk_sel')]=clgjk_nmt
                    data_nmt['lb'] = lb_nmt
                
                #data_nmt['binning'] = bins_nmt
                data_nmt['bpwf'] = wins = w.get_bandpower_windows().squeeze()
                w.write_to('mll_'+args.out_file.replace('.pkl','.fits'))
                #data_pkl['binning'] = bins
                
                all_data_pkl={}
                all_data_pkl['qml'] = data_pkl
                all_data_pkl['nmt'] = data_nmt
                with open(out_fname,"wb") as f:
                    #pkl.dump(data_pkl,f)
                    pkl.dump(all_data_pkl,f)

    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='qml_analyzer_gaia',
                    description='Compute QML and NMT spectra of gaia quasar',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--catalog',dest='catalog',action='store',type=str,required=True,help='Gaia Catalog') 
    parser.add_argument('--out_file',dest='out_file',action='store',type=str,required=True,help='Output file name')
    parser.add_argument('--mask', dest='mask',action='store',type=str,default='planck60',help='Mask type based on planck galactic mask or values of the selection function {planckfsky,selfuncfsky}')
    parser.add_argument('--noise_type',dest='noise_type',action='store',type=str,default='diagk_diagg',help='noise type to be processed separated by comma {diagk_diagg,corrk_diagg,diagk_corrg}')
    parser.add_argument('--catalog_path',dest='catalog_path',action='store',type=str,default='/Users/gfabbian/Work/quasar_gaia/catalog_20230406',help='GAIA catalog path')
    parser.add_argument('--lensing_path',dest='lensing_path',action='store',type=str,default='/Users/gfabbian/Work/PR4_variations',help='CMB lensing data path')
    parser.add_argument('--lensing_map',dest='lensing_map',action='store',type=str,default='PR4',help='Planck lensing map')
    parser.add_argument('--mask_path',dest='mask_path',action='store',type=str,default='/Users/gfabbian/Work/quasar_gaia',help='Masks path')
    parser.add_argument('--marginalize_dipole',dest='marginalize_dipole',action='store_true')
    parser.add_argument('--nodipole_mocks',dest='nodipole_mocks',action='store_true')
    parser.add_argument('--deproj_systematics',dest='systematics',action='store',type=str,default=None,help='List systeamtics template separated by commas {dust,star,mcs,m10,star_wise}')
    parser.add_argument('--nside_lr',dest='nside_lr',action='store',type=int,default=16,help='nside of QML maps') 
    parser.add_argument('--nside_hr',dest='nside_hr',action='store',type=int,default=256,help='nside of NMT maps')
    parser.add_argument('--nsims',dest='nsim',action='store',type=int,default=100,help='number of MC simulations')
    parser.add_argument('--n_galsplit',dest='n_galsplit',action='store',type=int,default=100,help='number of catalog splits used to compute galaxy splits and jackknives')
    parser.add_argument('--cl_theory',dest='cl_theory',action='store',type=str,default='/Users/gfabbian/Work/quasar_gaia/gaia-quasars-lss/notebooks/camb_thcl_ffp10.pkl',help='Theory cl')
    parser.add_argument('--binning',dest='binning',action='store',type=str,default='desi',help='bin spacing or desi for custom binning')
    parser.add_argument('-v', '--verbose',dest='verbose',action='store_true')

    args = parser.parse_args()
    main(args)