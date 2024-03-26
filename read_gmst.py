import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

file = '~/nas/home/BERK/global_mean_monthly/Land_and_Ocean_complete.txt'

df = pd.read_csv(file, comment = '%', header = None, delim_whitespace = True)

# use +/- two years for the 49-month moving average
start_year_idx = np.where(df[0] == 1898)[0][0]
end_year_idx = np.where(df[0] == 2017)[0][11]

raw_monthly = df.iloc[start_year_idx:end_year_idx+1, 2].values
# apply a 49-month moving average to raw_monthly
sm_monthly = np.convolve(raw_monthly, np.ones(49)/49, mode='valid')
pause = 1

gmst = np.zeros((116, 12))
for mm in range(12):
    
    gmst[:, mm] = sm_monthly[mm::12]

gmst
np.savetxt('gmst.txt', gmst)
pause = 1