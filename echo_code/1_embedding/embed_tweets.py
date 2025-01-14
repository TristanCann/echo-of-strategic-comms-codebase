import os
from tqdm import tqdm

from pathlib import Path
import sys
utils_folder = str(Path(__file__).parents[1]) # gets the parent folder (the repo folder) which contains the utils python file
sys.path.append(utils_folder)
from utils import *

from embed_texts import *

TWEET_FOLDER = ''
SAVE_FOLDER = ''
SAMPLE_TESTING = False

tweet_fnames = [f for f in os.listdir(TWEET_FOLDER) if f.startswith('climatechange1')]
tweet_fpaths = [TWEET_FOLDER + f for f in tweet_fnames]

if SAMPLE_TESTING:
    rng = np.random.default_rng(seed=26)
    inds = rng.choice(np.arange(len(tweet_fpaths)), 10, replace=False)
    tweet_fnames = np.array(tweet_fnames)[inds]
    tweet_fpaths = np.array(tweet_fpaths)[inds]

for fname, fpath in tqdm(zip(tweet_fnames, tweet_fpaths), total=len(tweet_fpaths)):
    df = load_tweet_file(fpath) # loads tweet text dataframe
    df['text'] = df['text'].apply(clean_tweet) # use function from utils to perfom tweet-specific cleaning
    embeddings = clean_embed(df.text.values) # generates embeddings for tweets
    
    dh = dh_string_from_tweet_fname(fname)
    with open(f'{SAVE_FOLDER}public_{dh}_embeddings.npy', 'wb') as f:
        np.save(f, embeddings)