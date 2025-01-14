import os
from pathlib import Path
import sys
utils_folder = str(Path(__file__).parents[1]) # gets the parent folder (the repo folder) which contains the utils python file
sys.path.append(utils_folder)
from utils import *

from embed_texts import *

PR_FOLDER = ''
SAVE_FOLDER = ''

pr_fnames = os.listdir(PR_FOLDER)
pr_fpaths = [PR_FOLDER + f for f in pr_fnames]

# combine all pr data into one dataframe
rows = []
for fname, fpath in zip(pr_fnames, pr_fpaths):
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()
        
    org = split_pr_fname(fname)[0] # split filename to extract org name
    
    # use known file structure
    date = lines[0] # date on line 0
    url = lines[1] # url on line 1
    title = lines[2] # title on line 2
    body = '\n'.join(lines[3:]) # body text is rest of file

    row = [] # collate row entry for dataframe
    row.append(fname)
    row.append(title)
    row.append(org)
    row.append(date) #date
    row.append(url) #url
    row.append(body) #body text
    
    rows.append(row)
    
# turn list of rows into dataframe
df = pd.DataFrame(rows, columns=['fname', 'title', 'org', 'date', 'url', 'body'])

# process dataframe
df['date'] = pd.to_datetime(df['date']) # convert string to datetime
df['text'] = df['title'] + "\n" + df['body'] # make new column combining title and body text

# embed combined title and body text, after sentencization
embeddings, sents = clean_embed(df.text.values, long_texts=True, sentence_means=True, return_sents=True, show_progress_bar=True, batch_size=32)
df['sents'] = list(sents)

# save df
save_path = SAVE_FOLDER + 'pr_info.json'
df.to_json(save_path)

# save embeddings
with open(f'{SAVE_FOLDER}pr_embeddings.npy', 'wb') as f:
	np.save(f,embeddings)