SELECT
gaia.ra, gaia.dec, gaia.source_id, sdss.ra AS ra_sdss, sdss.dec AS dec_sdss, sdss.z AS z_sdss, 
sdss.plate, sdss.mjd, sdss.fiberid, sdss.id, sdss.qso_id, sdss.zwarning, 
sdss.imatch, sdss.comp_boss, sdss.sector_ssr,
sdss.u_mag_sdss, sdss.g_mag_sdss, sdss.r_mag_sdss, sdss.i_mag_sdss, sdss.z_mag_sdss,
gaia.phot_g_mean_mag, gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag, gaia.phot_bp_n_obs, gaia.phot_rp_n_obs
FROM user_kstoreyf.eboss_quasars_slim AS sdss
LEFT JOIN gaiadr3.gaia_source AS gaia
ON 1 = CONTAINS(
   POINT(sdss.ra, sdss.dec),
   CIRCLE(gaia.ra, gaia.dec, 0.00028))
