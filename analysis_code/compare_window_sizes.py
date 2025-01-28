## This script compares the different window sizes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
parent_folder = str(Path(__file__).parents[1])

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=14)
arr_blue, arr_green, arr_red, arr_purple = np.array([[0.368, 0.506, 0.709]]), np.array([[0.560, 0.691, 0.194]]), np.array([[0.922, 0.385, 0.209]]), np.array([[0.528, 0.470, 0.701]])
blue, green, red, purple, white  = (0.368, 0.506, 0.709), (0.560, 0.691, 0.194), (0.922, 0.385, 0.209), (0.528, 0.470, 0.701), (1,1,1)

## Load the data.
df_1_1 = pd.read_json(f'{parent_folder}/data/pr_echo_1_1_0.700.json')
df_1_3 = pd.read_json(f'{parent_folder}/data/pr_echo_1_3_0.700.json')
df_1_7 = pd.read_json(f'{parent_folder}/data/pr_echo_1_7_0.700.json')
df_3_1 = pd.read_json(f'{parent_folder}/data/pr_echo_3_1_0.700.json')
df_3_3 = pd.read_json(f'{parent_folder}/data/pr_echo_3_3_0.700.json')
df_3_7 = pd.read_json(f'{parent_folder}/data/pr_echo_3_7_0.700.json')
df_7_1 = pd.read_json(f'{parent_folder}/data/pr_echo_7_1_0.700.json')
df_7_3 = pd.read_json(f'{parent_folder}/data/pr_echo_7_3_0.700.json')
df_7_7 = pd.read_json(f'{parent_folder}/data/pr_echo_7_7_0.700.json')

## Augment each df with some jitter to make the plots more readable.
np.random.seed(42)
jitter = np.random.uniform(low=-0.25,high=0.25,size=len(df_1_1))
df_1_1['jitter'] = jitter
df_1_3['jitter'] = jitter
df_1_7['jitter'] = jitter
df_3_1['jitter'] = jitter
df_3_3['jitter'] = jitter
df_3_7['jitter'] = jitter
df_7_1['jitter'] = jitter
df_7_3['jitter'] = jitter
df_7_7['jitter'] = jitter
for pr_to_show in ['350_20200218_bezos-climate-pledge.txt','greenpeace_20210809_ipcc-report-signals-decisive-moment-for-humanity-urgent-climate-action-needed.txt']:
	## Plot the values as bands.
	plt.figure(figsize=(7.5,4.5))
	plt.scatter(np.array([1]*(len(df_1_1)-1))+df_1_1[df_1_1.index != pr_to_show].jitter,df_1_1[df_1_1.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([2]*(len(df_1_1)-1))+df_1_3[df_1_3.index != pr_to_show].jitter,df_1_3[df_1_3.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([3]*(len(df_1_1)-1))+df_1_7[df_1_7.index != pr_to_show].jitter,df_1_7[df_1_7.index != pr_to_show].delta_raw,alpha=0.1,color=red)

	plt.scatter(np.array([4]*(len(df_3_1)-1))+df_3_1[df_3_1.index != pr_to_show].jitter,df_3_1[df_3_1.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([5]*(len(df_3_3)-1))+df_3_3[df_3_3.index != pr_to_show].jitter,df_3_3[df_3_3.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([6]*(len(df_3_7)-1))+df_3_7[df_3_7.index != pr_to_show].jitter,df_3_7[df_3_7.index != pr_to_show].delta_raw,alpha=0.1,color=red)

	plt.scatter(np.array([7]*(len(df_1_1)-1))+df_7_1[df_7_1.index != pr_to_show].jitter,df_7_1[df_7_1.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([8]*(len(df_1_1)-1))+df_7_3[df_7_3.index != pr_to_show].jitter,df_7_3[df_7_3.index != pr_to_show].delta_raw,alpha=0.1,color=red)
	plt.scatter(np.array([9]*(len(df_1_1)-1))+df_7_7[df_7_7.index != pr_to_show].jitter,df_7_7[df_7_7.index != pr_to_show].delta_raw,alpha=0.1,color=red)

	## Annotate a particular value in the plots.
	plt.scatter(1,df_1_1.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(2,df_1_3.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(3,df_1_7.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(4,df_3_1.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(5,df_3_3.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(6,df_3_7.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(7,df_7_1.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(8,df_7_3.loc[pr_to_show].delta_raw,color='black')
	plt.scatter(9,df_7_7.loc[pr_to_show].delta_raw,color='black')

	plt.ylim((-100000,100000))
	plt.xticks([1,2,3,4,5,6,7,8,9],labels=['(1,1)','(1,3)','(1,7)','(3,1)','(3,3)','(3,7)','(7,1)','(7,3)','(7,7)'])
	plt.ylabel(r'$\Delta_{raw}$')
	plt.xlabel('(Pre,Post) window size in days')
	plt.yscale('symlog')
	plt.savefig(f'figs/delta_raw_diff_windows_{pr_to_show}_highlight.pdf',bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(7.5,4.5))
	plt.scatter(np.array([1]*(len(df_1_1)-1))+df_1_1[df_1_1.index != pr_to_show].jitter,df_1_1[df_1_1.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([2]*(len(df_1_1)-1))+df_1_3[df_1_3.index != pr_to_show].jitter,df_1_3[df_1_3.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([3]*(len(df_1_1)-1))+df_1_7[df_1_7.index != pr_to_show].jitter,df_1_7[df_1_7.index != pr_to_show].delta_prop,alpha=0.1,color=red)

	plt.scatter(np.array([4]*(len(df_3_1)-1))+df_3_1[df_3_1.index != pr_to_show].jitter,df_3_1[df_3_1.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([5]*(len(df_3_3)-1))+df_3_3[df_3_3.index != pr_to_show].jitter,df_3_3[df_3_3.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([6]*(len(df_3_7)-1))+df_3_7[df_3_7.index != pr_to_show].jitter,df_3_7[df_3_7.index != pr_to_show].delta_prop,alpha=0.1,color=red)

	plt.scatter(np.array([7]*(len(df_1_1)-1))+df_7_1[df_7_1.index != pr_to_show].jitter,df_7_1[df_7_1.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([8]*(len(df_1_1)-1))+df_7_3[df_7_3.index != pr_to_show].jitter,df_7_3[df_7_3.index != pr_to_show].delta_prop,alpha=0.1,color=red)
	plt.scatter(np.array([9]*(len(df_1_1)-1))+df_7_7[df_7_7.index != pr_to_show].jitter,df_7_7[df_7_7.index != pr_to_show].delta_prop,alpha=0.1,color=red)

	## Annotate a particular value in the plots.
	plt.scatter(1,df_1_1.loc[pr_to_show].delta_prop,color='black') 
	plt.scatter(2,df_1_3.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(3,df_1_7.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(4,df_3_1.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(5,df_3_3.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(6,df_3_7.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(7,df_7_1.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(8,df_7_3.loc[pr_to_show].delta_prop,color='black')
	plt.scatter(9,df_7_7.loc[pr_to_show].delta_prop,color='black')

	plt.xticks([1,2,3,4,5,6,7,8,9],labels=['(1,1)','(1,3)','(1,7)','(3,1)','(3,3)','(3,7)','(7,1)','(7,3)','(7,7)'])
	plt.ylabel(r'$\Delta_{prop}$')
	plt.xlabel('(Pre,Post) window size in days')
	plt.savefig(f'figs/delta_prop_diff_windows_{pr_to_show}_highlight.pdf',bbox_inches='tight')
	plt.close()