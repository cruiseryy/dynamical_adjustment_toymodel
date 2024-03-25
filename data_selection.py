import numpy as np
import xarray as xr
from utilities import *
from time import time

# this script is to select data within the study region

# temporal pool range 1900-2015 (116 years)
# spatial pool T: [35N, 65N], [15W, 25E]
# spatial pool SLP: [25N, 90N], [60W, 50E]

suf = '.nc'
path_slp = '~/nas/home/20CR_V3/daily/'
var = 'prmsl'

# years * day in a year * lat * lon
slp = np.zeros((116, 365, 65, 160))
for yy in range(1900, 2016):
    t1 = time()
    file = path_slp + var + '.' + str(yy) + suf
    with xr.open_dataset(file) as ds:
        # manually selecting the study region 
        tmp1 = ds[var].values[:, 115:-1, :100]
        tmp2 = ds[var].values[:, 115:-1, 300:]
        tmp_slp = np.concatenate((tmp2, tmp1), axis=2)
        if is_leap(yy):
            tmp_slp = np.delete(tmp_slp, 59, axis=0)
        slp[yy-1900] = tmp_slp
    print('Year', yy, 'is done. Time:', time()-t1)
    pause = 1

path_temp = '~/nas/home/BERK/Complete_TMAX_Daily_LatLong1_'
temp = np.zeros((120, 365, 65, 160))
for yy in range(1900, 2020, 10):
    t1 = time()
    file = path_temp + str(yy) + suf
    with xr.open_dataset(file) as ds:
        tmp_temp = ds['temperature'].values[:, 115:, 120:280]
        cur = 0
        for ty in range(10):
            if is_leap(yy+ty):
                tmp_temp = np.delete(tmp_temp, cur + 59, axis=0)
            temp[yy-1900+ty] = tmp_temp[cur:cur+365, :, :]
            cur += 365
    print('Year', yy, 'is done. Time:', time()-t1)
temp = temp[:116]

# define the dimensions
years = np.arange(1900, 2016)  # 116 years
days = np.arange(1, 366)       # Assuming 365 days for simplification
lats = np.linspace(25.5, 90.5, 65, endpoint=False)  # Adjust as per your actual latitude range
lons = np.linspace(-59.5, 100.5, 160, endpoint=False)  # Adjust as per your actual longitude range

# create an xarray dataset
ds = xr.Dataset({
    'slp': (['year', 'day', 'lats', 'lons'], slp)
}, coords={
    'year': years,
    'day': days,
    'lats': lats,
    'lons': lons
})

# Save the Dataset as a NetCDF file
nc_file_path = 'slp.nc'
ds.to_netcdf(nc_file_path)

print(f'Saved SLP data to {nc_file_path}')

ds2 = xr.Dataset({
    'temp': (['year', 'day', 'lats', 'lons'], temp)
}, coords={
    'year': years,
    'day': days,
    'lats': lats,
    'lons': lons
})

# Save the Dataset as a NetCDF file
nc_file_path2 = 'temp.nc'
ds2.to_netcdf(nc_file_path2)

