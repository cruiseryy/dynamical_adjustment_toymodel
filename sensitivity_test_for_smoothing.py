import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

from utilities import *
from ipcc_cmaps import *

# temporal pool range 1900-2015 (116 years)
# data selection range: [25N, 90N], [60W, 100E]: for both T and SLP

dd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # days in each month
lats_mesh = np.linspace(25.5, 90.5, 65, endpoint=False) # 25N - 90N
lons_mesh = np.linspace(-59.5, 100.5, 160, endpoint=False) # 60W - 100E

temp_month = np.zeros((116, 12, 65, 160))
slp_month = np.zeros((116, 12, 65, 160))
with xr.open_dataset('data/temp.nc') as ds, xr.open_dataset('data/slp.nc') as ds2:
    cur = 0
    for mm in range(12):
        temp_month[:, mm, :, :] = np.mean(ds['temp'].values[:, cur:cur+dd[mm], :, :], axis = 1) 
        slp_month[:, mm, :, :] = np.mean(ds2['slp'].values[:, cur:cur+dd[mm], :, :], axis = 1)
        cur += dd[mm]   

# 4-year (49-month) smoothed GMST based on BERK monthly GMST data (sea-ice temperature inferred from air temp)
# refer to van, otto and other WWA work to justify the window size of 4 years (to remove ENSO related variability)
gmst_all = np.loadtxt('data/gmst.txt') 

# for the loess filte, use a 45-year window followed by a 2-year window following Terray et al. 2021
frac = [45./116, 2./116] 

# save the forced (low-freq) components for later analysis (detrending)
temp_f1 = np.zeros((116, 12, 65, 160))
temp_f2 = np.zeros((116, 12, 65, 160))
slp_f1 = np.zeros((116, 12, 65, 160))
slp_f2 = np.zeros((116, 12, 65, 160))

# for plotting
extent = [-25, 40, 30, 75]
cmap = get_ipcc_cmap('misc_seq')

# # for Tmax
# for mm in range(12):
#     # compare the R-squared and correlation coefficient (of residual time series) between the two smoothing methods
#     rsq_map = np.zeros((65, 160))
#     corr_map = np.zeros((65, 160))
#     gmst = gmst_all[:, mm].reshape(-1, 1)
#     t1 = time()
#     for i in range(65):
#         for j in range(160):

#             tmp_ts = temp_month[:, mm, i, j].reshape(-1, 1)
#             if math.isnan(np.mean(tmp_ts)):
#                 rsq_map[i, j] = np.nan
#                 corr_map[i, j] = np.nan
#                 temp_f1[:, mm, i, j] = np.nan
#                 temp_f2[:, mm, i, j] = np.nan
#                 continue
            
#             # smoothing using a loess filter
#             fit1, res1 = loess_smooth(tmp_ts, frac = frac)

#             # smoothing using a linear regression (4-year moving average GMST)
#             lm1 = LinearRegression()
#             lm1.fit(gmst, tmp_ts)
#             fit2 = lm1.predict(gmst)
#             res2 = tmp_ts - fit2

#             # calculate R^2 and correlation coefficient
#             lm2 = LinearRegression().fit(res1, res2)
#             rsq_map[i, j] = lm2.score(res1, res2)
#             corr_map[i, j] = np.corrcoef(res1.T, res2.T)[0, 1]

#             temp_f1[:, mm, i, j] = fit1.squeeze()
#             temp_f2[:, mm, i, j] = fit2.squeeze()

#     fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
#     plot_shading_map(ax[0], rsq_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap)
#     ax[0].set_title('R^2')
#     plot_shading_map(ax[1], corr_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap)
#     ax[1].set_title('Pearson\'s rho')
#     fig.savefig(f'test_figs/smoothing_compare_{mm:02d}.pdf')
#     plt.close(fig)

#     print('month %d, time elapsed: %.2f' % (mm, time() - t1))
#     print(f'rsq: {np.nanmean(rsq_map):.2f} [{np.nanmin(rsq_map):.2f}, {np.nanmax(rsq_map):.2f}], corr: {np.nanmean(corr_map):.2f} [{np.nanmin(corr_map):.2f}, {np.nanmax(corr_map):.2f}]')

# for SLP
for mm in range(12):
    # compare the R-squared and correlation coefficient (of residual time series) between the two smoothing methods
    rsq_map = np.zeros((65, 160))
    corr_map = np.zeros((65, 160))
    gmst = gmst_all[:, mm].reshape(-1, 1)
    t1 = time()
    for i in range(65):
        for j in range(160):

            tmp_ts = slp_month[:, mm, i, j].reshape(-1, 1)
            if math.isnan(np.mean(tmp_ts)):
                rsq_map[i, j] = np.nan
                corr_map[i, j] = np.nan
                temp_f1[:, mm, i, j] = np.nan
                temp_f2[:, mm, i, j] = np.nan
                continue
            
            # smoothing using a loess filter
            fit1, res1 = loess_smooth(tmp_ts, frac = frac)

            # smoothing using a linear regression (4-year moving average GMST)
            lm1 = LinearRegression()
            lm1.fit(gmst, tmp_ts)
            fit2 = lm1.predict(gmst)
            res2 = tmp_ts - fit2

            # calculate R^2 and correlation coefficient
            lm2 = LinearRegression().fit(res1, res2)
            rsq_map[i, j] = lm2.score(res1, res2)
            corr_map[i, j] = np.corrcoef(res1.T, res2.T)[0, 1]

            slp_f1[:, mm, i, j] = fit1.squeeze()
            slp_f2[:, mm, i, j] = fit2.squeeze()


    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_shading_map(ax[0], rsq_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap, sst_mask = False)
    ax[0].set_title('R^2')
    plot_shading_map(ax[1], corr_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap, sst_mask = False)
    ax[1].set_title('Pearson\'s rho')
    fig.savefig(f'test_figs/smoothing_compare_{mm:02d}_slp.pdf')
    plt.close(fig)

    print('month %d, time elapsed: %.2f' % (mm, time() - t1))
    print(f'rsq: {np.nanmean(rsq_map):.2f} [{np.nanmin(rsq_map):.2f}, {np.nanmax(rsq_map):.2f}], corr: {np.nanmean(corr_map):.2f} [{np.nanmin(corr_map):.2f}, {np.nanmax(corr_map):.2f}]')

# define the dimensions
years = np.arange(1900, 2016)  # 116 years
months = np.arange(1, 13)       # Assuming 365 days for simplification
lats = np.linspace(25.5, 90.5, 65, endpoint=False)  # Adjust as per your actual latitude range
lons = np.linspace(-59.5, 100.5, 160, endpoint=False)  # Adjust as per your actual longitude range

# ds = xr.Dataset({
#     'temp_f': (['year', 'month', 'lats', 'lons'], temp_f1)
# }, coords={
#     'year': years,
#     'month': months,
#     'lats': lats,
#     'lons': lons
# })
# nc_file_path = 'data/forced_temp_loess_smooth.nc'
# ds.to_netcdf(nc_file_path)

# ds2 = xr.Dataset({
#     'temp_f': (['year', 'months', 'lats', 'lons'], temp_f2)
# }, coords={
#     'year': years,
#     'month': months,
#     'lats': lats,
#     'lons': lons
# })
# nc_file_path2 = 'data/forced_temp_gmst_regression.nc'
# ds2.to_netcdf(nc_file_path2)

ds3 = xr.Dataset({
    'slp_f': (['year', 'month', 'lats', 'lons'], slp_f1)
}, coords={
    'year': years,
    'month': months,
    'lats': lats,
    'lons': lons
})
nc_file_path3 = 'data/forced_slp_loess_smooth.nc'
ds3.to_netcdf(nc_file_path3)

ds4 = xr.Dataset({
    'slp_f': (['year', 'months', 'lats', 'lons'], slp_f2)
}, coords={
    'year': years,
    'month': months,
    'lats': lats,
    'lons': lons
})
nc_file_path4 = 'data/forced_slp_gmst_regression.nc'
ds4.to_netcdf(nc_file_path4)