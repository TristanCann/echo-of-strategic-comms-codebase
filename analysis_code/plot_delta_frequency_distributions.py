## This script reads in the delta values for each press release and plots the distributions for delta raw and delta prop.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
parent_folder = str(Path(__file__).parents[1])

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=14)
arr_blue, arr_green, arr_red, arr_purple = np.array([[0.368, 0.506, 0.709]]), np.array([[0.560, 0.691, 0.194]]), np.array([[0.922, 0.385, 0.209]]), np.array([[0.528, 0.470, 0.701]])
blue, green, red, purple, white  = (0.368, 0.506, 0.709), (0.560, 0.691, 0.194), (0.922, 0.385, 0.209), (0.528, 0.470, 0.701), (1,1,1)

## Parameters.
PRESS_RELEASE_FILE = f'{parent_folder}/data/pr_echo_7_3_0.700.json'

## Read the press release data.
df_pr_data = pd.read_json(PRESS_RELEASE_FILE)

## Ensure data columns are in the right format.
df_pr_data = df_pr_data.astype({'delta_raw': 'float32', 'delta_prop':'float32'})

## Now we plot the histograms of the delta values.
plt.hist(df_pr_data.delta_raw,bins=50,color=red)
plt.ylabel('Frequency')
plt.xlabel(r'$\Delta_{raw}$')
plt.yscale('log')
plt.savefig('figs/Fig5a.pdf',bbox_inches='tight')
plt.close()

fig,ax1 = plt.subplots()
ax1.hist(df_pr_data.delta_prop,bins=50,color=red)
ax1.set_ylabel('Frequency')
ax1.set_xlabel(r'$\Delta_{prop}$')
ax1.set_yscale('log')

## Add an inset visualisation of how many press releases have no similar tweets.
ax2 = fig.add_axes([0.6,0.55,0.3,0.3])  ## [left,bottom,width,height]
ax2.axis('off')
ax2.pie([len(df_pr_data)-df_pr_data.delta_raw.isna().sum(),df_pr_data.delta_raw.isna().sum()],textprops=dict(fontsize=8),colors=[red,blue],autopct='%.1f%%')  

## Add the annotations to the pie chart manually.
ax1.annotate('PRs with similar tweets',xy=(0.075,1600),xytext=(0.075,1600),size=8)
ax1.annotate('PRs without similar tweets',xy=(0.075,75),xytext=(0.075,75),size=8)

plt.savefig('figs/Fig5b.pdf',bbox_inches='tight')
plt.close()
