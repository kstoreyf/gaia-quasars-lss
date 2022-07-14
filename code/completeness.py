import pandas as pd
mapFile = 'allsky_m10_hpx7.h5'
dfm10 = pd.read_hdf(mapFile, "data")
print('File %s loaded.' % (mapFile))



def selectionFunctionRADEC(ra,dec,G):
	
	import numpy as np
	
	#These are the best-fit value of the ten free parameters we optimised in our model:
	a,b,c,d,e,f,x,y,z,lim,sigma = [1.0154179774831278,
	 -0.008254847738351057,
	 0.6981959151433699,
	 -0.07503539255843136,
	 1.7491113533977052,
	 0.4541796235976577,
	 -0.06817682843336803,
	 1.5712714454917935,
	 -0.12236281756914291,
	 20.53035927443456,
	 7.82854815003104e-05]
	
	
	
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
	
	
	def selectionFunction(G,m10):
	    """ Predicts the completeness at magnitude G, given a value of M_10 read from a precomputed map. 
	    
	    Parameters
	    ----------
	    G:   nd.array
	         where to evaluate the function
	    m10: float
	         the value of M_10 in a given region
	
	    Returns
	    -------
	    sf(G) between 0 and 1. 
	    """
	    
	    predictedG0 = a*m10+b
	    if m10>lim:
	        predictedG0 = c*m10 + (a-c)*lim + b
	
	    predictedInvslope = x*m10+y
	    if m10>lim:
	        predictedInvslope = z*m10 + (x-z)*lim + y
	
	    predictedShape = d*m10+e
	    if m10>lim:
	        predictedShape = f*m10 + (d-f)*lim + e
	        
	    return sigmoid(G, predictedG0, predictedInvslope, predictedShape)
	
	#for coordinates (ra,dec) find the number of the healpix:
	import healpy as hp
	m10 = dfm10[ hp.ang2pix(128, ra, dec, lonlat=True,nest=True) ]
	return selectionFunction(G,m10)
	
	
	
##################################
# This is the part where we call the function at chosen coordinates.
	
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

l = np.arange(-180,180,1)
b = np.arange(-90,90,1)
l, b = np.meshgrid(l, b)
c = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')

#the map is in (ra,dec) so we need to convert:
ra = c.icrs.ra.degree.flatten() 
dec = c.icrs.dec.degree.flatten() 
completeness21 = [ selectionFunctionRADEC(rr,dd,21) for rr,dd in zip(ra,dec) ]

#and plot
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter( ra , dec , c=completeness21 , vmin=0 , vmax=1,s=1)
plt.colorbar(label='completeness at G=21')
plt.xlabel('ra (degrees)')
plt.ylabel('dec (degrees)')

plt.figure(2)
plt.scatter( l.flatten() , b.flatten() , c=completeness21 , vmin=0 , vmax=1,s=1)
plt.colorbar(label='completeness at G=21')
plt.xlim(180,-180)
plt.ylim(-90,90)
plt.xlabel('l (degrees)')
plt.ylabel('b (degrees)')
plt.show()