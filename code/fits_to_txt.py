import numpy as np
from astropy.table import Table


G_str = '20.0'
fn_txt = f'../data/quaia_G{G_str}_minimal_reformat.txt'
fn_gcat = f'../data/quaia_G{G_str}.fits'
tab = Table.read(fn_gcat)
print(tab.columns)

#columns = ['source_id', 'l', 'b', 'redshift_quaia', 'redshift_quaia_err', 'phot_g_mean_mag']
columns = ['l', 'b', 'redshift_quaia', 'redshift_quaia_err', 'phot_g_mean_mag']
#precisions = [18, 14, 14, 7, 7, 6]
precisions = [14, 14, 7, 7, 6]
widths = [18, 18, 8, 8, 7]

data = np.array([np.array(tab[col]) for col in columns]).T

#subset for testing
#data = data[np.random.choice(np.arange(len(data)), size=10, replace=False)]

header = ','.join(columns)
# fmts = [f'%{precisions[i]+3}.{precisions[i]}f' if columns[i]!='source_id' \
# 	else f'%{precisions[i]+3}d' for i in range(len(columns))]	
# fmts = [f'%.{precisions[i]}f' if columns[i]!='source_id' \
# 	else f'%d' for i in range(len(columns))]	
fmts = [f'%{widths[i]}.{precisions[i]}f' for i in range(len(columns))]	

print(fmts)
np.savetxt(fn_txt, data, 
	   fmt=fmts,
	   delimiter=' ', header=header)

