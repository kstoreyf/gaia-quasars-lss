import numpy as np
import pandas as pd

import healpy as hp


### Completeness model map functions

def get_completeness(ra, dec, gmag):
    """ Get the completeness for a a given value or list 
    of ra(s), dec(s), G-band magnitude(s)

    Parameters
    ----------
    ra : nd.array
        RA at which to evaluate the completeness model
    dec : nd.array
        dec at which to evaluate the completeness model
    gmag : nd.array
        G-band magnitude at which to evaluate the completeness model

    Returns
    -------
    nd.array
        Evaluation of the completeness model, in the range [0,1].
    """
    fn_comp = '../data/completeness_allsky_m10_hpx7.h5'
    dfm10 = pd.read_hdf(fn_comp, "data")

    fn_params = '../data/completeness_model_params.dat'
    model_params = np.loadtxt(fn_params)
    
    pixel_indices = hp.ang2pix(128, ra, dec, lonlat=True, nest=True)
    m10 = dfm10[pixel_indices]
    return selectionFunction(gmag, m10, model_params)


def sigmoid(G, G0, invslope, shape):
    """ Generalized sigmoid function 
    
    Parameters
    ----------
    G: nd.array
        where to evaluate the function
    G0: float
        inflection point
    invslope: float
        steepness of the linear part. Shallower for larger values
    shape: float
        if shape=1, model is the classical logistic function, 
        shape converges to zero, then the model is a Gompertz function.

    Returns
    -------
    f(G) evaluation of the model. 
    """
    delta = G - G0
    return 1 - (0.5 * (np.tanh(delta/invslope) + 1)) ** shape


def selectionFunction(G,m10,model_params):
    """ Predicts the completeness at magnitude G, given a value of M_10 read from a precomputed map. 
    
    Parameters
    ----------
    G:   nd.array
            where to evaluate the function
    m10: float
            the value of M_10 in a given region
    model_params: nd.array
            the stored parameters of the trained model

    Returns
    -------
    sf(G) between 0 and 1. 
    """
    a,b,c,d,e,f,x,y,z,lim,sigma = model_params
    
    def sigmoid_params_belowlim(m10_vals):
        pG0 = a*m10_vals+b
        pInvslope = x*m10_vals+y
        pShape = d*m10_vals+e
        return pG0, pInvslope, pShape

    def sigmoid_params_abovelim(m10_vals):
        pG0 = c*m10_vals + (a-c)*lim + b
        pInvslope = z*m10_vals + (x-z)*lim + y
        pShape = f*m10_vals + (d-f)*lim + e
        return pG0, pInvslope, pShape
    
    N_m10 = len(m10)
    predictedG0 = np.empty(N_m10)
    predictedInvslope = np.empty(N_m10)
    predictedShape = np.empty(N_m10)

    idx_above_lim = (m10 > lim)
    idx_below_lim = np.invert(idx_above_lim)
    predictedG0[idx_below_lim], predictedInvslope[idx_below_lim], \
            predictedShape[idx_below_lim] = sigmoid_params_belowlim(m10[idx_below_lim])
    predictedG0[idx_above_lim], predictedInvslope[idx_above_lim], \
            predictedShape[idx_above_lim] = sigmoid_params_abovelim(m10[idx_above_lim])
        
    return sigmoid(G, predictedG0, predictedInvslope, predictedShape)
