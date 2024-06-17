import numpy as np
from astropy.table import Table


G_str = '20.5'
tag = ''
#tag = '_minimal'
fn_txt = f'../data/quaia_G{G_str}{tag}.txt'
fn_gcat = f'../data/quaia_G{G_str}.fits'
tab = Table.read(fn_gcat)
print(tab.columns)

#columns = ['source_id', 'l', 'b', 'redshift_quaia', 'redshift_quaia_err', 'phot_g_mean_mag']
if 'minimal' in tag:
    columns = ['l', 'b', 'redshift_quaia', 'redshift_quaia_err', 'phot_g_mean_mag']
    precisions = [14, 14, 7, 7, 6]
    widths = [18, 18, 8, 8, 7]
    fmts = [f'%{widths[i]}.{precisions[i]}f' for i in range(len(columns))]
else:
    columns = list(tab.columns)
    #columns = columns[:2]
    print(columns)
    #columns = ['unwise_objid']
    precisions = []
    types = []
    for i in range(len(columns)):
        col = columns[i]
        print(col)
        print(type(tab[col][0]))
        print(tab[col][0])
        if isinstance(tab[col][0], str):
            precisions.append( len(max(tab[col], key=len)) )
            types.append('s')
        elif isinstance(tab[col][0], (int, np.integer)):
            precisions.append( len(str(np.max(tab[col]))) )
            types.append('d')
        else:
            precisions.append( len(str(np.max(tab[col]))) )
            types.append('f')
    
    # add one space more than needed to fit value
    widths = [p+1 for p in precisions]
    # increase by one to fit negative sign if column has
    widths = [widths[i]+1 if (np.min(precisions[i])<0) else widths[i] for i in range(len(columns))]
    # don't need that extra space for first col
    widths[0] -= 1
    # for some reason, adding too much space for this col, don't understand why...
    if 'unwise_objid' in columns:
        widths[columns.index('unwise_objid')] -= 1
    fmts = [f'%{widths[i]}.{precisions[i]}{types[i]}' for i in range(len(columns))]
        

#data = np.array([np.array(tab[col]) for col in columns]).T
tab = tab[columns]
tab['unwise_objid'] = np.char.decode(tab['unwise_objid'])
data = np.array(tab.as_array())
#subset for testing
#data = data[np.random.choice(np.arange(len(data)), size=10, replace=False)]

header = ','.join(columns)
# fmts = [f'%{precisions[i]+3}.{precisions[i]}f' if columns[i]!='source_id' \
# 	else f'%{precisions[i]+3}d' for i in range(len(columns))]	
# fmts = [f'%.{precisions[i]}f' if columns[i]!='source_id' \
# 	else f'%d' for i in range(len(columns))]	

print(columns)
print(fmts)
np.savetxt(fn_txt, data, 
	   fmt=fmts,
	   delimiter=' ', header=header)

