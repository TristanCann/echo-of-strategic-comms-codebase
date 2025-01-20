import os
import pandas as pd
import json
import code

from tqdm import tqdm

from pathlib import Path
import sys
utils_folder = str(Path(__file__).parents[1]) # gets the parent folder (the repo folder) which contains the utils python file
sys.path.append(utils_folder)
from utils import *

# standard run
SIM_DF_FOLDER = ''
PR_DF_FILEPATH = ''
SAVE_FOLDER = ''
SAMPLE = False

THRESHOLD = 0.7
WINDOW_PAIRS = [(1,1),(1,3),(1,7),(3,1),(3,3),(3,7),(7,1),(7,3),(7,7)]
INCLUDE_RELEASE_DAY_IN_POST = True

fnames = sorted(os.listdir(SIM_DF_FOLDER))

pr_df = pd.read_json(PR_DF_FILEPATH).set_index('fname').sort_values('date')

## Some of the columns left over here are unecessary and costly for memory and storage.
## body and title content should be duplicated in sents.
pr_df = pr_df.drop(labels=['body','title'],axis='columns')


# initialize new columns for press release dataframe
echo_colnames = ['max_sim','delta_raw', 'delta_prop', 'pre_raw', 'post_raw', 'pre_thresh', 'post_thresh'] # names of new columns to be added
echo_cols = pd.DataFrame(dtype=object, columns=echo_colnames) # specify object as dtype so sequences can be added as elements 
echo_cols.loc[:,['delta_raw', 'delta_prop']] = echo_cols.loc[:,['delta_raw', 'delta_prop']].astype(float) # change dtype for two float columns

## This step can be saved to the end, instead we can fill out echo cols, then use this trick to update at the end.
pr_df[echo_colnames] =  echo_cols # add new columns (as empty dataframe) to press release dataframe

## Make copies to calculate all window pairs in one go - avoid IO, comm slow downs.
echo_cols = [echo_cols.copy() for _ in WINDOW_PAIRS]  ## copy() should suffice here since they are empty dfs.

if SAMPLE:
    rng = np.random.default_rng(26)
    fnames = rng.choice(fnames, 50, replace=False)

pr_df_fnames = []

for fname in tqdm(fnames):
    
    pr_df_fname = fname[:-5]+'.txt' # change file extension to match fnames in pr dataframe
    pr_df_fnames.append(pr_df_fname)
    pr_date = pr_df.loc[pr_df_fname].date.date() # use this altered fname to query pr dataframe for this pr's date
    sim_df = pd.read_json(f'{SIM_DF_FOLDER}{fname}') # load the similarity dataframe

    ## Repeat for each window pair required.
    for j,(PRE_WINDOW, POST_WINDOW) in enumerate(WINDOW_PAIRS):

        # calculate the echo metric
        delta_raw, delta_prop, pre_thresh, post_thresh, pre_raw, post_raw = calculate_influence(sim_df, pr_date, threshold=THRESHOLD,
                                                              window_pre=PRE_WINDOW, window_post=POST_WINDOW,
                                                              include_release_day_in_post=INCLUDE_RELEASE_DAY_IN_POST, return_dfs=True)
    
        # explictly make array with object dtype - prevents error messages (as otherwise this is done implicitly)
        row = np.array([sim_df.sim.max(),delta_raw, delta_prop, list(pre_raw), list(post_raw), list(pre_thresh), list(post_thresh)], dtype=object) 

        echo_cols[j].loc[pr_df_fname, echo_colnames] = row


for i,(PRE_WINDOW,POST_WINDOW) in enumerate(WINDOW_PAIRS):
    ## This adds echo_cols to pr_df as intended at the start.
    save_df = pr_df.copy()
    save_df.update(echo_cols[i])

    ## Drop those press releases that shouldn't have a row in the file (those too close to the dataset limits).
    save_df = save_df.dropna(subset='pre_raw')

    save_df.to_json(f'{SAVE_FOLDER}batch_pr_echo_{str(int(PRE_WINDOW))}_{str(int(POST_WINDOW))}_{THRESHOLD:.3f}.json')