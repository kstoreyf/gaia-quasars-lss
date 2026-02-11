#!/usr/bin/env python3
"""
Fetch Gaia XP spectra for QUAIA sources that have spectra, in batches,
and append results to an HDF5 file with a clear structure.

Usage:
  python get_spectra.py --quaia ../data/quaia_G20.5.fits --output ../data/quaia_xp_spectra.h5
  python get_spectra.py --quaia ../data/quaia_G20.5.fits --output ../data/quaia_xp_spectra.h5 --batch-size 500
  python get_spectra.py -n 50  # test with first 50 sources
"""

import argparse
import time
from pathlib import Path

import numpy as np
import h5py
from astropy.table import Table
from astroquery.gaia import Gaia


# Default spectrum type: XP_SAMPLED gives flux vs wavelength; XP_CONTINUOUS gives coefficients
#DEFAULT_SPECTRUM_TYPE = "XP_CONTINUOUS"
DEFAULT_SPECTRUM_TYPE = "gaiaxpy_calibrated"
#DEFAULT_SPECTRUM_TYPE = "XP_SAMPLED"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_SLEEP_S = 5


def load_quaia(fn_quaia):
    """Load QUAIA catalog and return table."""
    tab = Table.read(fn_quaia)
    if "SOURCE_ID" in tab.colnames:
        tab.rename_column("SOURCE_ID", "source_id")
    return tab


def _gaia_bool_to_numpy(col):
    """Convert Gaia 't'/'f' or True/False to numpy bool."""
    arr = np.asarray(col)
    if arr.dtype.kind == "U" or arr.dtype.kind == "S":
        return arr == "t"
    return arr.astype(bool)


def get_quaia_has_xp(tab_quaia, fn_has_xp, upload_table_name="ids"):
    """
    Get table of QUAIA source_ids with has_xp_continuous and has_xp_sampled.
    If fn_has_xp exists, load from file; else query Gaia TAP and save.
    Returns Table with columns source_id, has_xp_continuous, has_xp_sampled.
    """
    fn_has_xp = Path(fn_has_xp)
    if fn_has_xp.exists():
        tab = Table.read(fn_has_xp)
        for old, new in [("SOURCE_ID", "source_id"), ("HAS_XP_CONTINUOUS", "has_xp_continuous"), ("HAS_XP_SAMPLED", "has_xp_sampled")]:
            if old in tab.colnames and old != new:
                tab.rename_column(old, new)
        return tab

    ids_tab = Table()
    ids_tab["source_id"] = tab_quaia["source_id"]

    adql = """
    SELECT g.source_id, g.has_xp_continuous, g.has_xp_sampled
    FROM gaiadr3.gaia_source AS g
    JOIN TAP_UPLOAD.{upload_table_name} AS u
      ON g.source_id = u.source_id
    """.format(
        upload_table_name=upload_table_name
    )

    job = Gaia.launch_job_async(
        adql,
        upload_resource=ids_tab,
        upload_table_name=upload_table_name,
    )
    tab = job.get_results()
    for old, new in [("SOURCE_ID", "source_id"), ("HAS_XP_CONTINUOUS", "has_xp_continuous"), ("HAS_XP_SAMPLED", "has_xp_sampled")]:
        if old in tab.colnames and old != new:
            tab.rename_column(old, new)
    for col in ("has_xp_continuous", "has_xp_sampled"):
        tab[col] = _gaia_bool_to_numpy(tab[col])

    fn_has_xp.parent.mkdir(parents=True, exist_ok=True)
    tab.write(fn_has_xp, overwrite=True)
    return tab


def get_quaia_source_ids_with_xp(tab_has_xp, spectrum_type):
    """
    Return list of source_ids that have XP spectra of the given type.
    spectrum_type: 'XP_SAMPLED', 'XP_CONTINUOUS', or 'gaiaxpy_calibrated' 
    (uses has_xp_sampled / has_xp_continuous).
    """
    if spectrum_type == "gaiaxpy_calibrated":
        # gaiaxpy.calibrate works with continuous coefficients, so use has_xp_continuous
        flag = "has_xp_continuous"
    else:
        flag = "has_xp_continuous" if spectrum_type == "XP_CONTINUOUS" else "has_xp_sampled"
    mask = tab_has_xp[flag]
    return list(tab_has_xp["source_id"][mask])


def create_empty_spectra_hdf5(fn_h5, spectrum_type=DEFAULT_SPECTRUM_TYPE, n_wave_bp=60, n_wave_rp=60):
    """
    Create an empty HDF5 file with resizable datasets for appending batches.
    Structure:
      /meta/source_id     (1D, int64, extendable)
      /spectra/wave_bp    (1D, set from first batch)
      /spectra/wave_rp    (1D, set from first batch)
      /spectra/flux_bp    (2D: n x n_wave, extendable both dims)
      /spectra/flux_rp    (2D: n x n_wave, extendable both dims)
    Wavelength and flux second dimension are set when the first batch is appended.
    """
    Path(fn_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fn_h5, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset(
            "source_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            compression="gzip",
        )
        meta.attrs["description"] = "Gaia source_id for each spectrum row"
        meta.attrs["n_total"] = 0

        spec = f.create_group("spectra")
        spec.create_dataset(
            "wave_bp",
            shape=(n_wave_bp,),
            maxshape=(None,),
            dtype=np.float64,
        )
        spec.create_dataset(
            "wave_rp",
            shape=(n_wave_rp,),
            maxshape=(None,),
            dtype=np.float64,
        )
        spec.create_dataset(
            "flux_bp",
            shape=(0, 0),
            maxshape=(None, None),
            dtype=np.float64,
            compression="gzip",
        )
        spec.create_dataset(
            "flux_rp",
            shape=(0, 0),
            maxshape=(None, None),
            dtype=np.float64,
            compression="gzip",
        )
        spec.attrs["spectrum_type"] = spectrum_type
        spec.attrs["description"] = "Gaia XP spectra (BP and RP); may be sampled or coefficients"

        f.attrs["created_by"] = "get_spectra.py"
        f.attrs["n_sources"] = 0


def _find_column(tab, patterns, exclude=None):
    """
    Find first column whose name (lowercase) contains all of patterns and none of exclude.
    patterns: list of substrings that must all be present
    exclude: list of substrings that must NOT be present (default: ['error','correlation'])
    """
    if exclude is None:
        exclude = ["error", "correlation"]
    for name in tab.colnames:
        n = name.lower()
        if all(p.lower() in n for p in patterns) and not any(e in n for e in exclude):
            return name
    return None


def _fetch_batch_gaiaxpy(batch_ids, save_file=False):
    """
    Fetch XP spectra via gaiaxpy.convert (handles archive fetch + conversion).
    Returns dict with source_id, flux_bp, flux_rp, wave_bp, wave_rp or None.

    gaiaxpy.convert returns a DataFrame with columns: source_id, xp (BP|RP), flux, flux_error.
    One row per band per source (2N rows for N sources). We pivot to get flux_bp, flux_rp.
    """
    #try:
    from gaiaxpy import convert
    import pandas as pd

    converted, sampling = convert(
        list(batch_ids),
        save_file=save_file,
    )
    if not isinstance(converted, pd.DataFrame) or len(converted) == 0:
        return None
    df = converted
    # gaiaxpy returns: source_id, xp (BP|RP), flux, flux_error
    if "xp" not in df.columns or "flux" not in df.columns:
        return None
    src_col = "source_id" if "source_id" in df.columns else next(
        (c for c in df.columns if "source" in c.lower() and "id" in c.lower()), None
    )
    if src_col is None:
        return None

    # Pivot: BP rows -> flux_bp, RP rows -> flux_rp (one row per source)
    bp_rows = df["xp"].str.upper() == "BP"
    rp_rows = df["xp"].str.upper() == "RP"
    if not (bp_rows.any() and rp_rows.any()):
        return None

    df_bp = df[bp_rows].sort_values(src_col)
    df_rp = df[rp_rows].sort_values(src_col)
    source_ids = np.asarray(df_bp[src_col], dtype=np.int64)
    if not np.array_equal(source_ids, np.asarray(df_rp[src_col], dtype=np.int64)):
        return None

    flux_bp = np.array([np.asarray(x) for x in df_bp["flux"]])
    flux_rp = np.array([np.asarray(x) for x in df_rp["flux"]])
    wave = np.asarray(sampling, dtype=np.float64) if sampling is not None else np.arange(
        flux_bp.shape[1], dtype=np.float64
    )
    return {
        "source_id": source_ids,
        "flux_bp": np.atleast_2d(flux_bp),
        "flux_rp": np.atleast_2d(flux_rp),
        "wave_bp": wave,
        "wave_rp": wave,
        "is_coefficients": False,
    }
    # except Exception:
    #     return None


def _fetch_batch_gaiaxpy_calibrated(batch_ids):
    """
    Fetch XP spectra via gaiaxpy.calibrate (handles archive fetch + calibration).
    Returns dict with source_id, flux_bp, flux_rp, wave_bp, wave_rp or None.

    gaiaxpy.calibrate returns a DataFrame with columns: source_id, flux, flux_error
    and a sampling wavelength array. The flux is a combined calibrated spectrum.
    Since the HDF5 structure expects separate BP and RP, we store the same flux in both.
    """
    from gaiaxpy import calibrate
    import pandas as pd

    calibrated_spectra, sampling = calibrate(list(batch_ids), save_file=False)
    if not isinstance(calibrated_spectra, pd.DataFrame) or len(calibrated_spectra) == 0:
        return None
    df = calibrated_spectra

    # gaiaxpy.calibrate returns: source_id, flux, flux_error
    if "flux" not in df.columns:
        return None
    src_col = "source_id" if "source_id" in df.columns else next(
        (c for c in df.columns if "source" in c.lower() and "id" in c.lower()), None
    )
    if src_col is None:
        return None

    source_ids = np.asarray(df[src_col], dtype=np.int64)
    flux = np.array([np.asarray(x) for x in df["flux"]])
    wave = np.asarray(sampling, dtype=np.float64) if sampling is not None else np.arange(
        flux.shape[1], dtype=np.float64
    )

    # Store the same calibrated flux in both BP and RP slots
    # since calibrate returns a combined spectrum
    return {
        "source_id": source_ids,
        "flux_bp": np.atleast_2d(flux),
        "flux_rp": np.atleast_2d(flux),
        "wave_bp": wave,
        "wave_rp": wave,
        "is_coefficients": False,
    }
    # except Exception:
    #     return None


def parse_datalink_spectra_to_arrays(datalink_result):
    """
    Parse Gaia load_data result into arrays we can write to HDF5.
    Gaia returns a dict: {product_key: [Table, ...], ...} with one table per source.
    Dynamically finds columns from the actual table (source_id, bp_coefficients, rp_coefficients, etc.).
    Returns dict with source_id, flux_bp, flux_rp (and optionally wave_*) or None.
    """
    try:
        # Gaia.load_data returns dict: {'XP_*-source_id.xml': [<Table length=1>], ...}
        if isinstance(datalink_result, dict):
            tables = []
            for key, val in datalink_result.items():
                if isinstance(val, (list, tuple)):
                    for t in val:
                        if isinstance(t, Table) and len(t) > 0:
                            tables.append(t)
                elif isinstance(val, Table) and len(val) > 0:
                    tables.append(val)
            if not tables:
                return None
            tab = Table.vstack(tables)
        elif hasattr(datalink_result, "to_table"):
            tab = datalink_result.to_table()
        elif hasattr(datalink_result, "get_results"):
            tab = datalink_result.get_results()
        else:
            tab = datalink_result
        if not isinstance(tab, Table) or len(tab) == 0:
            return None
    except Exception:
        return None

    # Dynamically find column names from the actual table
    src_col = None
    for c in tab.colnames:
        if c.lower().replace(" ", "_") in ("source_id", "sourceid"):
            src_col = c
            break
    if src_col is None:
        src_col = _find_column(tab, ["source", "id"])  # must have both to avoid solution_id
    if src_col is None:
        return None
    source_ids = np.asarray(tab[src_col], dtype=np.int64)

    # Find BP and RP coefficient columns (XP_CONTINUOUS) or flux columns (XP_SAMPLED)
    bp_coeff_col = _find_column(tab, ["bp", "coefficient"])
    rp_coeff_col = _find_column(tab, ["rp", "coefficient"])
    flux_bp_col = _find_column(tab, ["flux", "bp"], exclude=[])
    flux_rp_col = _find_column(tab, ["flux", "rp"], exclude=[])

    wave_bp = wave_rp = flux_bp = flux_rp = None

    # XP_SAMPLED: wavelength and flux columns
    for name in tab.colnames:
        name_lower = name.lower()
        col = tab[name]
        if hasattr(col, "shape") and len(col.shape) >= 1:
            if "wave" in name_lower and "bp" in name_lower:
                wave_bp = np.atleast_1d(np.asarray(col[0]))
            elif "wave" in name_lower and "rp" in name_lower:
                wave_rp = np.atleast_1d(np.asarray(col[0]))
            elif flux_bp_col and name == flux_bp_col:
                flux_bp = np.asarray(col)
            elif flux_rp_col and name == flux_rp_col:
                flux_rp = np.asarray(col)

    # XP_CONTINUOUS: bp_coefficients, rp_coefficients
    if bp_coeff_col and rp_coeff_col:
        bp_coeffs = np.array([np.asarray(r) for r in tab[bp_coeff_col]])
        rp_coeffs = np.array([np.asarray(r) for r in tab[rp_coeff_col]])
        # Handle variable-length: pad or use object array
        if bp_coeffs.dtype == object:
            lens_bp = [len(x) for x in bp_coeffs]
            lens_rp = [len(x) for x in rp_coeffs]
            n_bp, n_rp = max(lens_bp), max(lens_rp)
            bp_padded = np.zeros((len(bp_coeffs), n_bp), dtype=np.float64)
            rp_padded = np.zeros((len(rp_coeffs), n_rp), dtype=np.float64)
            for i, (b, r) in enumerate(zip(bp_coeffs, rp_coeffs)):
                bp_padded[i, : len(b)] = np.asarray(b)
                rp_padded[i, : len(r)] = np.asarray(r)
            bp_coeffs, rp_coeffs = bp_padded, rp_padded
        n_bp, n_rp = bp_coeffs.shape[1], rp_coeffs.shape[1]
        return {
            "source_id": source_ids,
            "flux_bp": bp_coeffs,
            "flux_rp": rp_coeffs,
            "n_wave_bp": n_bp,
            "n_wave_rp": n_rp,
            "is_coefficients": True,
        }

    # XP_SAMPLED with flux columns
    if flux_bp_col and flux_rp_col and flux_bp is not None and flux_rp is not None:
        n_bp = flux_bp.shape[1] if flux_bp.ndim > 1 else len(flux_bp)
        n_rp = flux_rp.shape[1] if flux_rp.ndim > 1 else len(flux_rp)
        if wave_bp is None:
            wave_bp = np.arange(n_bp, dtype=np.float64)
        if wave_rp is None:
            wave_rp = np.arange(n_rp, dtype=np.float64)
        return {
            "source_id": source_ids,
            "wave_bp": np.atleast_1d(wave_bp),
            "wave_rp": np.atleast_1d(wave_rp),
            "flux_bp": np.atleast_2d(flux_bp),
            "flux_rp": np.atleast_2d(flux_rp),
            "is_coefficients": False,
        }
    return None


def append_batch_to_hdf5(fn_h5, batch_data, is_first_batch=False):
    """
    Append one batch of spectrum data to the HDF5 file.
    batch_data: dict from parse_datalink_spectra_to_arrays (source_id, flux_bp, flux_rp, etc.)
    """
    if batch_data is None:
        return 0
    n_add = len(batch_data["source_id"])
    if n_add == 0:
        return 0

    with h5py.File(fn_h5, "a") as f:
        meta = f["meta"]
        spec = f["spectra"]
        n_current = meta["source_id"].shape[0]

        meta["source_id"].resize(n_current + n_add, axis=0)
        meta["source_id"][n_current:] = batch_data["source_id"]
        meta.attrs["n_total"] = n_current + n_add

        flux_bp = np.atleast_2d(batch_data["flux_bp"])
        flux_rp = np.atleast_2d(batch_data["flux_rp"])
        n_bp, n_rp = flux_bp.shape[1], flux_rp.shape[1]

        if is_first_batch:
            spec["wave_bp"].resize((n_bp,))
            spec["wave_rp"].resize((n_rp,))
            if "wave_bp" in batch_data:
                spec["wave_bp"][:] = batch_data["wave_bp"]
                spec["wave_rp"][:] = batch_data["wave_rp"]
            else:
                spec["wave_bp"][:] = np.arange(n_bp, dtype=np.float64)
                spec["wave_rp"][:] = np.arange(n_rp, dtype=np.float64)

        # Resize flux datasets and write
        spec["flux_bp"].resize(n_current + n_add, axis=0)
        spec["flux_rp"].resize(n_current + n_add, axis=0)
        if spec["flux_bp"].shape[1] != n_bp:
            spec["flux_bp"].resize(n_bp, axis=1)
        if spec["flux_rp"].shape[1] != n_rp:
            spec["flux_rp"].resize(n_rp, axis=1)
        spec["flux_bp"][n_current:] = flux_bp
        spec["flux_rp"][n_current:] = flux_rp

        f.attrs["n_sources"] = n_current + n_add

    return n_add


def run(
    fn_quaia,
    fn_output,
    batch_size=DEFAULT_BATCH_SIZE,
    spectrum_type=DEFAULT_SPECTRUM_TYPE,
    sleep_s=DEFAULT_SLEEP_S,
    skip_existing=True,
    limit=None,
):
    """
    Main pipeline: load QUAIA, find sources with XP, create HDF5, fetch in batches, append.
    """
    fn_quaia = Path(fn_quaia)
    fn_output = Path(fn_output)
    if not fn_quaia.exists():
        raise FileNotFoundError(f"QUAIA catalog not found: {fn_quaia}")

    print(f"Loading QUAIA from {fn_quaia}")
    tab_quaia = load_quaia(fn_quaia)
    n_quaia = len(tab_quaia)
    print(f"  QUAIA has {n_quaia} sources")

    fn_has_xp = fn_quaia.parent / ("has_xp_" + fn_quaia.name)
    if fn_has_xp.exists():
        print(f"Loading XP flags from {fn_has_xp}")
    else:
        print("Querying Gaia TAP for XP flags (source_id, has_xp_continuous, has_xp_sampled)...")
    tab_has_xp = get_quaia_has_xp(tab_quaia, fn_has_xp)
    source_ids_with_xp = get_quaia_source_ids_with_xp(tab_has_xp, spectrum_type)
    n_with_xp = len(source_ids_with_xp)
    print(f"  {n_with_xp} QUAIA sources have XP spectra ({100 * n_with_xp / n_quaia:.1f}%)")

    if n_with_xp == 0:
        print("Nothing to fetch. Exiting.")
        return

    # Track how many already exist if we're skipping
    n_existing = 0
    if fn_output.exists() and skip_existing:
        with h5py.File(fn_output, "r") as f:
            done_ids = set(f["meta"]["source_id"][:])
            n_existing = len(done_ids)
        source_ids_with_xp = [s for s in source_ids_with_xp if s not in done_ids]
        n_with_xp = len(source_ids_with_xp)
        print(f"  After skipping existing ({n_existing} already in file): {n_with_xp} left to fetch")

    if limit is not None:
        # If limit is set, ensure we don't exceed it total
        # So if limit=100 and 2 already exist, only add 98 more
        if n_existing >= limit:
            print(f"  Limit {limit} already reached ({n_existing} spectra in file). Nothing to add.")
            source_ids_with_xp = []
        else:
            max_to_add = limit - n_existing
            source_ids_with_xp = source_ids_with_xp[:max_to_add]
        n_with_xp = len(source_ids_with_xp)
        if n_with_xp > 0:
            print(f"  Limited to {n_with_xp} new sources (will have {n_existing + n_with_xp} total, limit={limit})")

    if n_with_xp == 0:
        print("All spectra already in file. Exiting.")
        return

    if not fn_output.exists():
        print(f"Creating empty HDF5 at {fn_output}")
        create_empty_spectra_hdf5(fn_output, spectrum_type=spectrum_type)
    else:
        print(f"Appending to existing HDF5 at {fn_output}")

    # Start total timing
    total_start_time = time.time()
    
    n_batches = (n_with_xp + batch_size - 1) // batch_size
    first_batch = True
    for i in range(0, n_with_xp, batch_size):
        batch = source_ids_with_xp[i : i + batch_size]
        batch_num = i // batch_size + 1
        
        # Start batch timing
        batch_start_time = time.time()
        print(f"Batch {batch_num}/{n_batches} ({len(batch)} sources)...", end=" ", flush=True)

        try:
            parsed = None
            if spectrum_type == "gaiaxpy_calibrated":
                parsed = _fetch_batch_gaiaxpy_calibrated(batch)
                # Do not fall back to Gaia.load_data: retrieval_type must be a Gaia product
                # (e.g. XP_CONTINUOUS, XP_SAMPLED), not "gaiaxpy_calibrated".
            elif spectrum_type == "XP_CONTINUOUS":
                parsed = _fetch_batch_gaiaxpy(batch)
            if parsed is None and spectrum_type in ("XP_CONTINUOUS", "XP_SAMPLED"):
                datalink = Gaia.load_data(
                    ids=batch,
                    data_release="Gaia DR3",
                    retrieval_type=spectrum_type,
                )
                parsed = parse_datalink_spectra_to_arrays(datalink)
            if parsed is not None:
                n_added = append_batch_to_hdf5(fn_output, parsed, is_first_batch=first_batch)
                if first_batch:
                    first_batch = False
                
                # Calculate and print batch time
                batch_time = time.time() - batch_start_time
                print(f"appended {n_added} spectra. (batch time: {batch_time:.2f}s)")
            else:
                batch_time = time.time() - batch_start_time
                if spectrum_type == "gaiaxpy_calibrated":
                    print(f"gaiaxpy.calibrate failed for this batch (check gaiaxpy install and archive access) (batch time: {batch_time:.2f}s)")
                else:
                    print(f"(parser could not extract arrays; check spectrum_type and Gaia product format) (batch time: {batch_time:.2f}s)")
        except Exception as e:
            batch_time = time.time() - batch_start_time
            print(f"Error: {e} (batch time: {batch_time:.2f}s)")

        if batch_num < n_batches:
            time.sleep(sleep_s)

    # Calculate and print total time
    total_time = time.time() - total_start_time
    with h5py.File(fn_output, "r") as f:
        total = f.attrs.get("n_sources", 0)
    print(f"Done. Total spectra in {fn_output}: {total}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Gaia XP spectra for QUAIA sources with spectra; write to HDF5 in batches."
    )
    parser.add_argument(
        "--quaia",
        type=str,
        default="../data/quaia_G20.5.fits",
        help="Path to QUAIA catalog FITS",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file (default: quaia_xp_spectra_sampled.h5 or quaia_xp_spectra_continuous.h5 by spectrum type)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sources per Gaia load_data request (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--spectrum-type",
        type=str,
        default=DEFAULT_SPECTRUM_TYPE,
        choices=["XP_SAMPLED", "XP_CONTINUOUS", "gaiaxpy_calibrated"],
        help="Gaia XP product type (XP_SAMPLED, XP_CONTINUOUS, or gaiaxpy_calibrated)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_S,
        help=f"Seconds to sleep between batches (default {DEFAULT_SLEEP_S})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip source_ids already present in output HDF5",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit to first N sources with spectra (for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        if args.spectrum_type == "XP_SAMPLED":
            suffix = "sampled"
        elif args.spectrum_type == "gaiaxpy_calibrated":
            suffix = "calibrated"
        elif args.spectrum_type == "XP_CONTINUOUS":
            suffix = "continuous"
        else:
            raise ValueError(f"Invalid spectrum type: {args.spectrum_type}")
        #args.output = f"../data/quaia_xp_spectra_{suffix}.h5"
        args.output = f"../data/quaia_x_dr16_prop_xp_spectra_{suffix}.h5"
    run(
        fn_quaia=args.quaia,
        fn_output=args.output,
        batch_size=args.batch_size,
        spectrum_type=args.spectrum_type,
        sleep_s=args.sleep,
        skip_existing=args.skip_existing,
        limit=args.limit,
    )
