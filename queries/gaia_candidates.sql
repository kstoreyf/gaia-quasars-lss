SELECT
g.source_id, g.ra, g.dec,  g.l, g.b, 
g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
g.phot_g_n_obs, g.phot_bp_n_obs, g.phot_rp_n_obs,
q.redshift_qsoc, q.redshift_qsoc_lower, q.redshift_qsoc_upper, q.zscore_qsoc, q.flags_qsoc,
g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
g.parallax, g.parallax_error
FROM gaiadr3.gaia_source as g
JOIN
gaiadr3.qso_candidates as q
ON q.source_id = g.source_id
