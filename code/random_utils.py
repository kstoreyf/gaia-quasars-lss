from astropy.table import Table

import utils
import generate_random


def get_tabs_subsampled(df_rconfigs):
    tabs_r = []
    for i in range(len(df_rconfigs)):
        tag_rand = ''

        if df_rconfigs['dust'][i]:
            NSIDE_dustmap = int(df_rconfigs['NSIDE_dustmap'][i])
            tag_rand += f'_dust{NSIDE_dustmap}'
        if df_rconfigs['completeness'][i]:
            tag_rand += f'_completeness{gmag_comp}'
        if df_rconfigs['stardens'][i]:
            NSIDE_starmap = int(df_rconfigs['NSIDE_starmap'][i])
            tag_rand += f'_stardens{NSIDE_starmap}'
        if df_rconfigs['stardust'][i]:
            NSIDE_stardustmap = int(df_rconfigs['NSIDE_stardustmap'][i])
            tag_rand += f'_stardust{NSIDE_stardustmap}'
        fac_rand = df_rconfigs['fac_rand'][i]
        tab_r = utils.load_table(f'../data/randoms/random{tag_rand}_{fac_rand}x.fits')
        tabs_r.append(tab_r)
    return tabs_r

def get_tabs_masked(df_rconfigs, ra_data, dec_data):

    NSIDE_masks = 64
    R = 3.1

    tabs_rmask = []
    tabs_dmask = []

    for i in range(len(df_rconfigs)):
        b_max, NSIDE_dustmap, Av_max = None, None, None
        mask_plane, mask_mcs, mask_dust = False, False, False

        if df_rconfigs['maskplane'][i]:
            b_max = df_rconfigs['b_max'][i]
            mask_plane = True
        if df_rconfigs['maskmcs'][i]:
            mask_mcs = True
        if df_rconfigs['maskdust'][i]:
            NSIDE_dustmap = int(df_rconfigs['NSIDE_dustmap'][i])
            Av_max = df_rconfigs['Av_max'][i]
            mask_dust = True

        tabs_r = get_tabs_subsampled(df_rconfigs)
            
        fn_dustmap = f'../data/dustmap_Avmean_NSIDE{NSIDE_dustmap}.npy'
        ra_rand, dec_rand = tabs_r[i]['ra'], tabs_r[i]['dec']
        ra_rmask, dec_rmask = generate_random.get_mask_indices(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                                        mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                        mask_dust=mask_dust,
                                        fn_dustmap=fn_dustmap, b_max=b_max,
                                        Av_max=Av_max, R=R)
        tabs_rmask.append( Table([ra_rmask, dec_rmask], names=['ra', 'dec']) )

        # now apply same masks to data
        ra_dmask, dec_dmask = generate_random.get_mask_indices(ra_data, dec_data, NSIDE_masks, NSIDE_dustmap,
                                        mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                        mask_dust=mask_dust,
                                        fn_dustmap=fn_dustmap, b_max=b_max,
                                        Av_max=Av_max, R=R)
        tabs_dmask.append( Table([ra_dmask, dec_dmask], names=['ra', 'dec']) )
        
    stitles, mtitles = [], []
    for i in range(len(df_rconfigs)):
        subsample_title, mask_title = get_title(df_rconfigs.iloc[i])
        stitles.append(subsample_title)
        mtitles.append(mask_title)
    df_rconfigs['subsample_title'] = stitles
    df_rconfigs['mask_title'] = mtitles
        
    return tabs_dmask, tabs_rmask


def get_idx_arrs_masked(df_rconfigs, ra_rand, dec_rand, ra_data, dec_data):

    idx_arrs_masks_rand = []
    idx_arrs_masks_data = []

    NSIDE_masks = 64
    R = 3.1

    for i in range(len(df_rconfigs)):
        b_max, NSIDE_dustmap, Av_max = None, None, None
        mask_plane, mask_mcs, mask_dust = False, False, False

        if df_rconfigs['maskplane'][i]:
            b_max = df_rconfigs['b_max'][i]
            mask_plane = True
        if df_rconfigs['maskmcs'][i]:
            mask_mcs = True
        if df_rconfigs['maskdust'][i]:
            NSIDE_dustmap = int(df_rconfigs['NSIDE_dustmap'][i])
            Av_max = df_rconfigs['Av_max'][i]
            mask_dust = True
            
        fn_dustmap = f'../data/dustmap_Avmean_NSIDE{NSIDE_dustmap}.npy'
        idx_rmask = generate_random.get_mask_indices(ra_rand, dec_rand, NSIDE_masks, NSIDE_dustmap,
                                        mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                        mask_dust=mask_dust,
                                        fn_dustmap=fn_dustmap, b_max=b_max,
                                        Av_max=Av_max, R=R)
        idx_arrs_masks_rand.append(idx_rmask)

        # now apply same masks to data
        idx_dmask = generate_random.get_mask_indices(ra_data, dec_data, NSIDE_masks, NSIDE_dustmap,
                                        mask_plane=mask_plane, mask_mcs=mask_mcs, 
                                        mask_dust=mask_dust,
                                        fn_dustmap=fn_dustmap, b_max=b_max,
                                        Av_max=Av_max, R=R)
        idx_arrs_masks_data.append(idx_dmask)
        
    mtitles = []
    for i in range(len(df_rconfigs)):
        mask_title = get_mask_title(df_rconfigs.iloc[i])
        mtitles.append(mask_title)
    df_rconfigs['mask_title'] = mtitles
        
    return idx_arrs_masks_data, idx_arrs_masks_rand



def get_title(df_row):
    if df_row['dust'] or df_row['completeness'] or df_row['stardens'] or df_row['stardust']:
        subsample_title = 'Subsample by'
    else:
        subsample_title = 'Uniform '
        
    if df_row['dust']:
        NSIDE_dustmap = int(df_row['NSIDE_dustmap'])
        subsample_title += f' dust (NSIDE={NSIDE_dustmap}),'
    if df_row['completeness']:
        subsample_title += ' completeness,'
    if df_row['stardens']:
        subsample_title += ' stellar density,'
    if df_row['stardust']:
        subsample_title += ' dust + stellar density fit,'
    subsample_title = subsample_title[:-1]+'.'
    
    if df_row['maskplane'] or df_row['maskmcs'] or df_row['maskdust']:
        mask_title = 'Masked by'
    else:
        mask_title = ' '
        
    if df_row['maskplane']:
        b_max = df_row['b_max']
        mask_title += f' plane (b_max={b_max}),'
    if df_row['maskmcs']:
        mask_title += f' MCs,'
    if df_row['maskdust']:
        NSIDE_dustmap = int(df_row['NSIDE_dustmap'])
        Av_max = df_row['Av_max']
        mask_title += f' dust (NSIDE={NSIDE_dustmap}, Av_max={Av_max}),'
    mask_title = mask_title[:-1]+'.'

    return subsample_title, mask_title


def get_mask_title(df_row):
    if df_row['maskplane'] or df_row['maskmcs'] or df_row['maskdust']:
        mask_title = 'Masked by'
    else:
        mask_title = 'No masking'

    if df_row['maskplane']:
        b_max = df_row['b_max']
        mask_title += f' plane (b_max={b_max}),'
    if df_row['maskmcs']:
        mask_title += f' MCs,'
    if df_row['maskdust']:
        NSIDE_dustmap = int(df_row['NSIDE_dustmap'])
        Av_max = df_row['Av_max']
        mask_title += f' dust (NSIDE={NSIDE_dustmap}, Av_max={Av_max}),'
    mask_title = mask_title[:-1]

    return mask_title