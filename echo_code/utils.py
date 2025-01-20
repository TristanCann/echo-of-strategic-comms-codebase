import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import zipfile
import re

def dayhour_strings_from_date(date, include_overflow_hours=True):
    '''
    Input: datetime date object
    Output: list of date-hour strings for use in tweet loading
    Optionally include hour either side of date 
    '''
    date_dt = datetime.combine(date, datetime.min.time()) # initializes time to midnight

    # get all hours to shift midnight (00) by to cover day's range
    if include_overflow_hours:
        hour_deltas = np.arange(-1,25) # -1 to 24 inclusive
    else:
        hour_deltas = np.arange(0,24) # 0 to 23 inclusive

    dayhour_dt = [date_dt + timedelta(hours=int(i)) for i in hour_deltas]
    dayhour_s = [dt.strftime('%Y%m%d%H') for dt in dayhour_dt]

    return dayhour_s


def dayhour_strings_from_dates(dates, include_overflow_hours=True):
    dayhour_s = []
    for date in dates:
        dayhour_s.extend(dayhour_strings_from_date(date, include_overflow_hours))
    return list(np.unique(dayhour_s)) # unique because for adjacent dates there will be duplicates when include_overflow_hours=True


def tweet_fname_from_dh_string(dh):
    '''
    Input: day hour string in format %Y%m%d%H
    Output: filename for tweets corresponding to the given day hour string
    '''
    return f'climatechange1_text_{dh}.zip'


def split_pr_fname(fname, orgs=['avaaz', 'nrdc', 'sierra_club', 'wwf', 'public_citizen', 'nwf', 'climate_reality_project', 'greenpeace', 'oxfam', '350']):
    org_match = re.match(f'(?:{"|".join(orgs)})', fname) # find match (at start of fname string) of any of the org strings
    org = org_match.group() # get org string from re match
    i = org_match.span()[1] # index of first character after matched string (i.e. first dividing '_')
    datestring = fname[i+1 : i+9] # datestring of known fixed length
    # pr_name = fname[i+10:-4] # get rest of fname (except extension)
    pr_name = ".".join(fname.split('.')[:-1])[i+10:] # get rest of fname (remove extension by splitting on .)

    return [org, datestring, pr_name]


def dh_string_from_tweet_fname(fname):
    '''
    Takes tweet df zip fname (e.g. climatechange1_text_2020110213.zip)
    Returns just dh string
    '''
    if fname.startswith('climatechange1_text_') and fname.endswith('.zip'):
        return fname[-14:-4]


def load_tweet_file(tweet_path):
    with zipfile.ZipFile(tweet_path, 'r') as archive: # open zip archive
        for file in archive.namelist(): # our structure: only 1 file in zip archive
            with archive.open(file) as f:
                df = pd.read_csv(f, delimiter='\t', keep_default_na=False, na_values=[], names=['timestamp', 'text', 'id', '_'], dtype={'id': str, 'text': str}).drop(columns='_').set_index('id') # load the csv tweet file as a pandas dataframe
    df = df.fillna('None')
    return df


def clean_tweet(tweet_text):
    '''
    Tweet-specific cleaning:
    1) remove t.co hyperlinks
    2) remove \x08 strings that appear in our twitter data (weird encoding at some point, probably)
    '''
    tweet_text = re.sub(r"https?:\/\/t.co\S+", "", tweet_text)
    tweet_text = re.sub(r"\x08", " ", tweet_text)
    return tweet_text


def clean_text(text):
    '''
    Apply basic regex cleaning to a generic text (used before sentencization for long texts)
    '''
    regex_sub_pairs = [
        (r"(\t|\n|\r)", " "), # remove newline, tab, carriage return characters
        (r"(\s\s+)", ' ') # remove additional spaces
    ]
    for pattern, repl in regex_sub_pairs:
        text = re.sub(pattern, repl, text)
    return text


def flatten(nested_list):
    '''
    Takes list of lists (nested)
    Returns flattened list with inds (i.e. which index element in lists did new elements originally belong to)
    '''
    lens = [len(l) for l in nested_list] # get length of each list element in nested_list
    raw_inds = np.arange(len(nested_list)).reshape(-1,1) # index each element of nested_list
    flattened_list = [item for sublist in nested_list for item in sublist] # flatten nested_list
    inds = np.repeat(raw_inds, lens) # the index of the list in nested_list the corresponding element in flattened_list belonged to
    return flattened_list, inds


def unflatten(flattened_list, inds):
    '''
    Takes a list and re-groups elements based on original placement, inds
    '''
    raw_inds = np.unique(inds)
    unflattened_list = [[el for i, el in enumerate(flattened_list) if inds[i] == x] for x in raw_inds]
    return unflattened_list


def unflatten_array(flattened_array, inds):
    '''
    unflatten algorithm optimised for arrays
    '''
    raw_inds = np.unique(inds)
    unflattened_list = [flattened_array[np.where(inds==x)[0]] for x in raw_inds]
    return unflattened_list


def calculate_influence(sim_df, pr_date, threshold,
                        window_pre=7, window_post=3, 
                        include_release_day_in_post=True, return_dfs=True):
    '''
    From a dataframe of similarities (where rows are tweets, each df corresponding to a press release),
    calculate the 'influence' or 'echo' of the press release, given the threshold and window parameters.
    Outputs both the standard and proportional echo metric.

    sim_df:
    pandas dataframe for a given press release
    each row corresponds to a tweet in that press release's 'window' - 7 days either side of the press release date
    must contain at least the following columns:
        sim: similarity of press release to tweet
        date: datetime date of the tweet
    
    pr_date: datetime.date() object - the date of publication of the press release
    threshold: the cosine similarity value at or above which a pr-tweet pair are considered sufficiently similar
    window_pre: the no. days in the pre window
        (e.g. 1 would include only the day before the press release date)
    window_post: the no. days in the post window
        by default, press release day included in post window, i.e. 1 includes only the press release day by default
        if include_release_day_in_post is false, 1 includes only the day after the press release day
    '''
    ## Note this check is no longer strictly required since the duplicates should have been removed already.
    ## Keeping in place for now to avoid any possibility of calculation with duplicates.
    sim_df = sim_df.loc[~sim_df.index.duplicated(keep='first')] # keep only unique tweet ids
    daily_count = sim_df.date.value_counts().sort_index() # get daily count of all tweets before further filtering
    
    # get dates in pre window, calculated from press release date and window size
    pre_dates = sorted([pd.Timestamp(pr_date - timedelta(days=i)) for i in range(window_pre, 0, -1)]) # stops at 1, so includes as far as day before pr
    
    # get dates in post window - by default includes press release date 
    if include_release_day_in_post:
        post_dates = sorted([pd.Timestamp(pr_date + timedelta(days=i)) for i in range(0, window_post)])
    else:
        post_dates = sorted([pd.Timestamp(pr_date + timedelta(days=i)) for i in range(1, window_post+1)])

    # get the total tweet count in each of the pre and post windows
    ## Note the use of get with a default here as some audiences will have days in which there are no tweets.
    pre_raw = np.array([daily_count.get(d,default=0) for d in pre_dates])
    post_raw = np.array([daily_count.get(d,default=0) for d in post_dates])

    thresh_df = sim_df.loc[sim_df.sim >= threshold] # keep only sufficiently similar texts

    n_t = thresh_df.date.value_counts().reindex(daily_count.index, fill_value=0) # count the number of similar texts on each day in the window

    # the number of tweets above the threshold in the pre and post windows
    pre_thresh = np.array([n_t.get(d,default=0) for d in pre_dates])
    post_thresh = np.array([n_t.get(d,default=0) for d in post_dates])

    ## Add a check if there are no similar tweets - here we want to return NaNs instead of 0s for the echo scores.
    ## This is mainly a problem with some of the less receptive audiences, but the behaviour should be accounted for so
    ## we can distinguish between no response and no change.
    if sum(pre_thresh) == 0 and sum(post_thresh)==0:
        if return_dfs:
            return np.nan, np.nan, [0]*window_pre, [0]*window_post, pre_raw, post_raw
        else:
            return np.nan, np.nan

    # the standard "raw" echo metric
    echo = np.nanmean(post_thresh) - np.nanmean(pre_thresh) # difference in means between post and pre windows

    # calculate the proportional echo metric
    post = np.nanmean(post_thresh/post_raw)
    pre = np.nanmean(pre_thresh/pre_raw)
    ## Replace nans in post or pre - if all the values are nan, returned value will be nan.
    if np.isnan(post) and not np.isnan(pre):  ## and here should be unnecessary with thresh checks but included for safety.
        post = 0
    if np.isnan(pre) and not np.isnan(post):
        pre = 0

    echo_prop = post - pre

    if return_dfs:
        return echo, echo_prop, pre_thresh, post_thresh, pre_raw, post_raw
    else:
        return echo, echo_prop