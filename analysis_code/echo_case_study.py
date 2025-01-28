## This script analyses the delta values observed for the preferred window sizes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import datetime
import matplotlib
from pathlib import Path
parent_folder = str(Path(__file__).parents[1])

## Parameters.
PRESS_RELEASE_FILE = f'{parent_folder}/data/pr_echo_7_3_0.700.json'

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=14)
arr_blue, arr_green, arr_red, arr_purple = np.array([[0.368, 0.506, 0.709]]), np.array([[0.560, 0.691, 0.194]]), np.array([[0.922, 0.385, 0.209]]), np.array([[0.528, 0.470, 0.701]])
blue, green, red, purple, white  = (0.368, 0.506, 0.709), (0.560, 0.691, 0.194), (0.922, 0.385, 0.209), (0.528, 0.470, 0.701), (1,1,1)

## Load the data
df_pr_data = pd.read_json(PRESS_RELEASE_FILE)

## Ensure the relevant columns are in the right format.
df_pr_data = df_pr_data.astype({'delta_raw': 'float32', 'delta_prop':'float32'})

## Load the timeseries of daily tweet activity.
df_day_counts = pd.read_json(f'{parent_folder}/data/tweet_counts_day.json')

## Plot a scatter of due date and influence for each delta.
fig,axs = plt.subplots(3,1,figsize=(16,8))
axs[0].scatter(df_pr_data.dropna(subset=['delta_raw'])['date'],df_pr_data.dropna(subset=['delta_raw'])['delta_raw'],alpha=0.3,color=red)
axs[0].set_ylabel(r'$\Delta_{raw}$')
axs[0].axhline(y=0, color='silver')
axs[0].set_xlim(datetime.date(year=2019,month=11,day=1),datetime.date(year=2021,month=11,day=1))

axs[1].scatter(df_pr_data.dropna(subset=['delta_prop'])['date'],df_pr_data.dropna(subset=['delta_prop'])['delta_prop'],alpha=0.3,color=red)
axs[1].set_ylabel(r'$\Delta_{prop}$')
axs[1].axhline(y=0, color='silver')
axs[1].set_xlim(datetime.date(year=2019,month=11,day=1),datetime.date(year=2021,month=11,day=1))

## Plot the tweet counts.
axs[2].plot(df_day_counts.index,df_day_counts['count'].values,color=blue)
axs[2].set_ylabel('Tweets per day')
axs[2].set_xlim(datetime.date(year=2019,month=11,day=1),datetime.date(year=2021,month=11,day=1))
axs[2].set_ylim(bottom=0)

for i in [0,2]:
	axs[i].get_yaxis().set_major_formatter(
		matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  ## Format the y axis labels to include commas.

plt.savefig('figs/Fig4.pdf',bbox_inches='tight')
plt.close()

## Calculate the correlation between delta raw and delta prop
corr = scipy.stats.pearsonr(df_pr_data.dropna(subset=['delta_raw'])['delta_raw'],df_pr_data.dropna(subset=['delta_raw'])['delta_prop'])
print('Pearson R raw vs prop:')
print(corr)

## Scatter delta_raw against delta_prop
plt.scatter(df_pr_data['delta_raw'],df_pr_data['delta_prop'],alpha=0.3,color=red)
plt.xlabel(r'$\Delta_{raw}$')
plt.ylabel(r'$\Delta_{prop}$')
ax = plt.gca()
ax.get_xaxis().set_major_formatter(
matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  ## Format the y axis labels to include commas.

## Add x=0 and y=0 lines
ax.hlines(y=0,xmin=-5000,xmax=23000,color='black',alpha=0.1)
ax.vlines(x=0,ymin=-0.025,ymax=0.125,color='black',alpha=0.1)
ax.set_ylim(-0.025,0.125)
ax.set_xlim(-5000,23000)
ax.annotate(r"Pearson's $R=0.811,p<0.01$",(7500,-0.015))  ## These statistics are known from previous runs, and printed to terminal to verify.

plt.savefig('figs/Fig6.pdf',bbox_inches='tight')