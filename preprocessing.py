import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from ipcc_cmaps import *

# temporal pool range 1900-2015 (116 years)
# spatial pool T: [35N, 65N], [15W, 25E]
# spatial pool SLP: [25N, 90N], [60W, 50E]

dd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # days in each month
lats_mesh = np.linspace(25.5, 90.5, 65, endpoint=False) # 25N - 90N
lons_mesh = np.linspace(-59.5, 100.5, 160, endpoint=False) # 50W - 100E

temp_month = np.zeros((116, 12, 65, 160))
with xr.open_dataset('temp.nc') as ds:
    cur = 0
    for mm in range(12):
        temp_month[:, mm, :, :] = np.mean(ds['temp'].values[:, cur:cur+dd[mm], :, :], axis = 1) 
        cur += dd[mm]   



test_time_series = temp_month[:, 0, 10, 100]
gmst = np.loadtxt('gmst.txt')

fit1, res1 = loess_smooth(test_time_series, frac = [45./116, 2./116])

lm = LinearRegression()
lm.fit(gmst, test_time_series)
fit2 = lm.predict(gmst)
res2 = test_time_series - fit2

pause = 1
fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize=(10, 10))
ax[0].plot(test_time_series)
ax[0].plot(fit1)
ax[1].plot(test_time_series)
ax[1].plot(fit2)
ax[2].plot(res1)
ax[3].plot(res2)
fig.savefig('test.pdf')
pause = 1

for i in range(65):
    for j in range(110):
        temp_month[:, :, i, j] = temp_month[:, :, i, j] - np.mean(temp_month[:, :, i, j])


# for map plotting
# mean_temp = np.mean(temp_month, axis = (0, 1))
# extent = [-25, 40, 30, 75]
# fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
# plot_shading_map(ax, mean_temp, lons_mesh, lats_mesh, extent = extent, cmap = get_ipcc_cmap('temp_div'))
# fig.savefig('test2.pdf')