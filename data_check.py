import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 20th century reanalysis V3

path = '~/nas/home/20CR_V3/daily/'
suf = 'nc'

var = ['prmsl']
for tv in var:
    for yy in range(1806, 2015):
        file = path + tv + '.' + str(yy) + '.' + suf
        print(file)
        with xr.open_dataset(file) as ds:
            keys = ds.keys()
            print(keys)
            pause = 1


# BERK: the Berkeley Earth Surface Tempearture 

path = '~/nas/home/BERK/Complete_TMAX_Daily_LatLong1_'
suf = '.nc'
for yy in range(1900, 2011, 10):
    file = path + str(yy) + suf
    print(file)
    with xr.open_dataset(file) as ds:
        keys = ds.keys()
        print(keys)
        pause = 1
    