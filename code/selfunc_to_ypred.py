import healpy as hp
import numpy as np 

import maps
import utils
import selection_function_map as sf

    
def main():
    G_max = 20.5
    tag_cat = '_zsplit2bin1'
    tag_sel = ''
    NSIDE = 64

    fn_gaia = f'../data/quaia_G{G_max}{tag_cat}.fits' 
    fn_selfunc = f"../data/maps/selection_function_NSIDE{NSIDE}_G{G_max}{tag_cat}{tag_sel}.fits"
    fn_ypred = f"../data/maps/y_pred_selection_function_NSIDE{NSIDE}_G{G_max}{tag_cat}{tag_sel}.fits"

    map_prob = hp.read_map(fn_selfunc)

    map_names = ['dust', 'stars', 'm10', 'mcs', 'unwise', 'unwisescan']

    print("Loading data", flush=True)
    tab_gaia = utils.load_table(fn_gaia)

    print("Making QSO map", flush=True)
    maps_forsel = sf.load_maps(NSIDE, map_names)
    map_nqso_data, _ = maps.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)

    print("Making ypred map", flush=True)
    map_ypred = map_probability_to_expected(map_prob, map_nqso_data, map_names, maps_forsel)

    hp.write_map(fn_ypred, map_ypred, overwrite=False)
    print(f"Saved ypred map to {fn_ypred}!", flush=True)


def map_expected_to_probability_orig(map_expected, map_true, map_names, maps_forsel):
        idx_clean = np.full(len(map_expected), True)
        for map_name, map in zip(map_names, maps_forsel):
            if map_name=='dust':
                idx_map = map < 0.03
            elif map_name=='stars':
                idx_map = map < 15
            elif map_name=='m10':
                idx_map = map > 21
            elif map_name=='mcs':
                idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be redundant w stars
            idx_clean = idx_clean & idx_map
        print("Number of clean healpixels:", np.sum(idx_clean), f"(Total: {len(map_expected)})")
        nqso_clean = np.mean(map_true[idx_clean])
        # 2 standard deviations above should mean that most, if not all, values are below 1
        nqso_max = nqso_clean + 1*np.std(map_true[idx_clean])
        map_prob = map_expected / nqso_max
        #map_prob[map_prob>1.0] = 1.0
        #assert np.all(map_prob <= 1.0) and np.all(map_prob >= 0.0), "Probabilities must be <=1 and >=0!"
        return map_prob


def map_probability_to_expected(map_prob, map_true, map_names, maps_forsel):
        idx_clean = np.full(len(map_prob), True)
        for map_name, map in zip(map_names, maps_forsel):
            if map_name=='dust':
                idx_map = map < 0.03
            elif map_name=='stars':
                idx_map = map < 15
            elif map_name=='m10':
                idx_map = map > 21
            elif map_name=='mcs':
                idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be redundant w stars
            idx_clean = idx_clean & idx_map
        print("Number of clean healpixels:", np.sum(idx_clean), f"(Total: {len(map_prob)})")

        nqso_clean = np.mean(map_true[idx_clean])
        # orig was 1 stdev
        nqso_max = nqso_clean + 1*np.std(map_true[idx_clean])

        map_expected = map_prob * nqso_max

        return map_expected

if __name__=='__main__':
    main()