import numpy as np
from astropy.table import Table


fn_txt = '../data/quaia_G20.5_minimal.txt'
fn_gcat = '../data/quaia_G20.5.fits'
tab = Table.read(fn_gcat)
print(tab.columns)

columns = ['source_id', 'l', 'b', 'redshift_quaia', 'redshift_quaia_err', 'phot_g_mean_mag']
precisions = [18, 14, 14, 7, 7, 6]

data = np.array([np.array(tab[col]) for col in columns]).T

header = ','.join(columns)
fmts = [f'%{precisions[i]+3}.{precisions[i]}f' if columns[i]!='source_id' \
	else f'%{precisions[i]+3}d' for i in range(len(columns))]	

np.savetxt(fn_txt, data, 
	   fmt=fmts,
	   delimiter=' ', header=header)

