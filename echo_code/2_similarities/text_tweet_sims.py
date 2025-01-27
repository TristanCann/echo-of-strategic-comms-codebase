import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from datetime import datetime, timedelta
import gc
import os

from pathlib import Path
import sys
utils_folder = str(Path(__file__).parents[1]) # gets the parent folder (the repo folder) which contains the utils python file
sys.path.append(utils_folder)
print(utils_folder)
from utils import *

### Manual Variables ###

PR_FILEPATH = ''
PR_EMBED_FILEPATH = ''
TWEET_EMBEDDINGS_FOLDER = ''
TWEET_FOLDER = ''
SIM_SAVE_FOLDER = ''

WINDOW = (7,7)

# valid range for pr date windows
# these dates are the extremes of our Twitter collection
TWEET_RANGE = (datetime(2019, 11,  1).date(), datetime(2021, 10, 31).date())

FORCE_RECALCULATE = True

# this function is used as a 'cache' so tweet files are not loaded unneccessarily often (a file is loaded once and kept in memory until it is no longer in the time window of interest)
def update_tweet_df(tweet_df, dh_strings):
    # get list of dh strings corresponding to files not currently loaded
    new_dh_strings = [dh for dh in dh_strings if dh not in tweet_df.file_dh.unique()]
    if len(new_dh_strings) == 0:
        return tweet_df
    
    tweet_df = tweet_df.loc[tweet_df.file_dh.isin(dh_strings)] # remove unneeded rows

    new_dfs = []
    for dh in new_dh_strings:
        fname = f'{TWEET_FOLDER}{tweet_fname_from_dh_string(dh)}'
        
        try:
            df = load_tweet_file(fname)

            #df.drop(labels=['text'],axis='columns')
        except FileNotFoundError:
            print(f'file for the hour {dh} does not exist')
            pass

        df['file_dh'] = dh # the dh used to load this tweet file
        df['sim'] = pd.Series(dtype=float) # empty float column for sim values
        df['timestamp'] = pd.to_datetime(df.timestamp) # convert timestamp to datetime
        df['date'] = df.timestamp.dt.date # get datetime date from timestamp column

        new_dfs.append(df)

    new_tweet_df = pd.concat(new_dfs)
    del new_dfs
    gc.collect()
    
    tweet_df = pd.concat([tweet_df, new_tweet_df])
    del new_tweet_df
    
    gc.collect()
    return tweet_df

### program ### 

## load and filter dataframe of press releases
pr_df = pd.read_json(PR_FILEPATH).set_index('fname') # fname is our unique identifier for each press release
len0 = len(pr_df)

with open(PR_EMBED_FILEPATH,'rb') as f:
    pr_embeds = np.load(f)

pr_df['embeddings'] = list(pr_embeds)

pr_df = pr_df.sort_values(['date','org'])

# filter prs - remove those with 7-day windows falling outside
pr_df = pr_df.loc[ ( (pr_df.date - timedelta(days=WINDOW[0])).dt.date >= TWEET_RANGE[0] ) & ( (pr_df.date + timedelta(days=WINDOW[1])).dt.date <= TWEET_RANGE[1] ) ]
len1 = len(pr_df)
print(f'removed {len0 - len1} out-of-bounds press releases')

already_calculated = [f[:-5] for f in os.listdir(SIM_SAVE_FOLDER) if f.endswith('.json')] # get list of similarity files already present in save folder
if isinstance(FORCE_RECALCULATE, bool): # if we're blanket recalculating all or nothing (i.e. True or False)
    if FORCE_RECALCULATE is False: # if we're not forcing any recalculation 
        pr_df = pr_df.loc[~pr_df.index.str[:-4].isin(already_calculated)] # remove all files that have been calculated already from the working dataframe
elif isinstance(FORCE_RECALCULATE, list): # if we're specifying files to recalculate
    a = len(already_calculated)
    already_calculated = [f for f in already_calculated if f + '.json' not in FORCE_RECALCULATE] # don't remove json filenames that are in FORCE_RECALCULATE from the working dataframe
    b = len(already_calculated)
    print(f'{a-b} excluded')
    pr_df = pr_df.loc[~pr_df.index.str[:-4].isin(already_calculated)] # remove all already calculated except those in FORCE_RECALCULATE

len2 = len(pr_df)
print(f'skipping {len1 - len2} press releases with similarities already calculated')

## initialize empty objects
tweet_df = pd.DataFrame(columns=['timestamp', 'text', 'file_dh', 'date', 'sim']) # dataframe
pr_window_dates = [] # initialise empty var

embeddings = {}

## iterate through press releases
for pr_fname, row in tqdm(pr_df.iterrows(), total=len(pr_df)):
    pr_date = row['date'].date() # press release date
    pr_window_dates = [pr_date + timedelta(days=int(i)) for i in np.arange(-WINDOW[0], WINDOW[1]+1)] # dates in window about press release date

    # if either first or last date is on the respective extreme of the tweet database window
    if pr_window_dates[0] == TWEET_RANGE[0] or pr_window_dates[-1] == TWEET_RANGE[1]:
        include_overflow_hours = False # if a date in the range falls on the first or last tweet day, don't include overflow hours
    else:
        include_overflow_hours = True

    dh_strings = dayhour_strings_from_dates(pr_window_dates, include_overflow_hours) # calculate dh strings
    tweet_df = update_tweet_df(tweet_df, dh_strings) # does nothing if the dates match those in tweet_df

    # removes unneeded dict elements
    for dh in list(embeddings.keys()):
        if dh not in dh_strings:
            del embeddings[dh]

    dh_sims = []
    for dh in dh_strings:
        if dh not in embeddings: # load embedding if not already cached
            embedding_fname = f'{TWEET_EMBEDDINGS_FOLDER}public_{dh}_embeddings.npy' # get fname from dh
            embeddings[dh] = np.load(embedding_fname) # load numpy array
        dh_embeddings = embeddings[dh] # load embeddings for that dh
        
        pr_embedding = row['embeddings'].reshape(1,-1) # get pr embedding
        dh_sims.append(cosine_similarity(pr_embedding, dh_embeddings).ravel()) # calculate similarities for this dh
    tweet_df['sim'] = np.concatenate(dh_sims) # combine all dh file sims together and add sims as dataframe column

    # clear any duplicate rows emerging from the df updating.
    out = tweet_df[~tweet_df.index.duplicated()].drop(labels=['text'],axis='columns')

    # filter out rows not in window dates - this filters out the unneeded overflow rows (loaded in case of timezone weirdness)
    out.loc[out.date.isin(pr_window_dates)].to_json(f'{SIM_SAVE_FOLDER}{pr_fname[:-4]}.json')
