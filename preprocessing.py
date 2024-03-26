import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from ipcc_cmaps import *
import math

# temporal pool range 1900-2015 (116 years)
# data selection range: [25N, 90N], [60W, 100E]: for both T and SLP
# study region: [35, 70N], [-10, 30E]

dd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # days in each month

with xr.open_dataset('data/temp.nc') as ds:
    temp_raw = ds['temp'].values
with xr.open_dataset('data/forced_temp_loess_smooth.nc') as ds:
    temp_f1 = ds['temp_f'].values
with xr.open_dataset('data/forced_temp_gmst_regression.nc') as ds:
    temp_f2 = ds['temp_f'].values

temp_anom1 = np.zeros(temp_raw.shape)
temp_anom2 = np.zeros(temp_raw.shape)

for i in range(65):
    for j in range(160):

        if math.isnan(np.mean(temp_raw[:, :, i, j], axis = (0, 1))):
            continue

        cur = 0
        for mm in range(12):
            temp_anom1[:, cur:cur+dd[mm], i, j] = temp_raw[:, cur:cur+dd[mm], i, j] - temp_f1[:, mm, i, j][:, None]
            temp_anom2[:, cur:cur+dd[mm], i, j] = temp_raw[:, cur:cur+dd[mm], i, j] - temp_f2[:, mm, i, j][:, None]
            cur += dd[mm]   

# for plotting
lats_mesh = np.linspace(25.5, 90.5, 65, endpoint=False) # 25N - 90N
lons_mesh = np.linspace(-59.5, 100.5, 160, endpoint=False) # 60W - 100E
extent = [-25, 40, 30, 75]
cmap = get_ipcc_cmap('prcp_div')

fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
plot_shading_map(ax[0], np.nanmean(temp_anom1, axis = (0, 1)), lons_mesh, lats_mesh, extent, cmap)
ax[0].set_title('Loess smoothed')
plot_shading_map(ax[1], np.nanmean(temp_anom2, axis = (0, 1)), lons_mesh, lats_mesh, extent, cmap)
ax[1].set_title('GMST regression')
fig.savefig('temp_anom_comparison_map.pdf')
plt.close(fig)


# demo script for plotting a density scatter plot of many sample points
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

temp_avg1 = np.nanmean(temp_anom1[:, :, 10:45, 50:90], axis = (2, 3)).squeeze()
temp_avg2 = np.nanmean(temp_anom2[:, :, 10:45, 50:90], axis = (2, 3)).squeeze()
pause = 1

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = 'scatter_density')
density_scatter = ax.scatter_density(temp_avg1, temp_avg2, cmap=white_viridis)
fig.colorbar(density_scatter, label = 'density')
plt.axline([0, 0], [1, 1], zorder = 0, color = 'black')
plt.xlabel('Loess smoothed')
plt.ylabel('GMST regression')
plt.savefig('temp_anom_comparison.pdf')
plt.close()

