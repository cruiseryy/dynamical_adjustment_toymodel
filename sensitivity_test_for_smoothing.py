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
lons_mesh = np.linspace(-59.5, 100.5, 160, endpoint=False) # 50W - 100E

temp_month = np.zeros((116, 12, 65, 160))
with xr.open_dataset('data/temp.nc') as ds:
    cur = 0
    for mm in range(12):
        temp_month[:, mm, :, :] = np.mean(ds['temp'].values[:, cur:cur+dd[mm], :, :], axis = 1) 
        cur += dd[mm]   

# this is 4-year (49-month) smoothed GMST based on BERK monthly GMST data (sea-ice temperature inferred from air temp)
# refer to van, otto and other WWA work to justify the window size of 4 years (to remove ENSO related variability)
gmst_all = np.loadtxt('data/gmst.txt') 
gmst = gmst_all[:, 0].reshape(-1, 1)

# for the loess filte, use a 45-year window followed by a 2-year window following Terray et al. 2021
frac = [45./116, 2./116] 

rsq_map = np.zeros((65, 160))
corr_map = np.zeros((65, 160))
for i in range(65):
    for j in range(160):
        t1 = time()
        tmp_ts = temp_month[:, 0, i, j].reshape(-1, 1)
        if math.isnan(np.mean(tmp_ts)):
            rsq_map[i, j] = np.nan
            corr_map[i, j] = np.nan
            continue
        pause = 1

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

        print('{}, {}: {:.2f} seconds'.format(i, j, time() - t1))

extent = [-25, 40, 30, 75]
cmap = get_ipcc_cmap('misc_seq')
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
plot_shading_map(ax[0], rsq_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap)
ax[0].set_title('R^2')
plot_shading_map(ax[1], corr_map, lons_mesh, lats_mesh, extent = extent, cmap = cmap)
ax[1].set_title('Pearson\'s rho')
fig.savefig('test_figs/smoothing_compare.pdf')


# # this is for compare the two ways of smoothing at a randomly selected pixel
# test_time_series = temp_month[:, 0, 10, 100].reshape(-1, 1)
# fit1, res1 = loess_smooth(test_time_series, frac = [45./116, 2./116])

# lm = LinearRegression()
# lm.fit(gmst, test_time_series)
# fit2 = lm.predict(gmst)
# res2 = test_time_series - fit2

# fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize=(10, 10))
# ax[0].plot(test_time_series)
# ax[0].plot(fit1)
# ax[1].plot(test_time_series)
# ax[1].plot(fit2)
# ax[2].plot(res1)
# ax[3].plot(res2)
# fig.savefig('test_figs/test.pdf')
# lm2 = LinearRegression().fit(res1, res2)
# R_sqaured = lm2.score(res1, res2)
# print('R^2: {:.2f}'.format(R_sqaured))
# print('corrcoef: {:.2f}'.format(np.corrcoef(res1.T, res2.T)[0, 1]))

# pause = 1
