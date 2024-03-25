import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import numpy as np 
from scipy.interpolate import splrep, BSpline
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, ElasticNet, LinearRegression
import matplotlib.patches as mpatches


def plot_shading_map(ax, data, lons, lats, extent, cmap):
    base = ax.contourf(lons, lats, data, levels = 21, transform=ccrs.PlateCarree(), cmap = cmap, extend='both')
    # ax.coastlines(resolution='10m', color='grey', linestyle='-', alpha=1)
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k', facecolor='whitesmoke')
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.plot([-10, 30, 30, -10, -10], [35, 35, 70, 70, 35],
         color='black', linewidth=1,
         transform=ccrs.PlateCarree(), zorder = 101)
    plt.colorbar(base, orientation='vertical', pad=0.02)
    
    return

def plot_contour_map(ax, data, lats, lons, 
             extent = [-30, 50, 25, 90]):
    ax.coastlines
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.contour(lons, lats, data, transform=ccrs.PlateCarree())
    return

def is_leap(yy):
    if yy % 4 == 0:
        if yy % 100 == 0:
            if yy % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def spline_smooth(y, s = 50):
    xx = np.arange(len(y))
    tck = splrep(xx, y, s = s)
    y_smoothed = BSpline(*tck)(xx)
    return y_smoothed, y - y_smoothed

def loess_smooth(y, frac):
    y = y.flatten()
    y0 = y
    xx = np.arange(len(y))
    for f in frac:
        fit = sm.nonparametric.lowess(y, xx, frac = f)
        y = fit[:, 1]
    res = y0 - fit[:, 1]
    return fit[:, 1].reshape(-1, 1), res.reshape(-1, 1)

def mv_avg_smooth(y, window = 5):
    y0 = y
    half_wind = window // 2
    y = np.convolve(y, np.ones(window)/window, mode='valid')
    return y, y0[half_wind:-half_wind] - y

def slp_distance(X0, XX):

    return

