### imports ###
import numpy as np
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer

# load utils

from pathlib import Path
import sys
utils_folder = str(Path(__file__).parents[1]) # gets the parent folder (the repo folder) which contains the utils python file
sys.path.append(utils_folder)
from utils import *

# manual variables

NLP = spacy.load("en_core_web_sm")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    
# method functions

def clean(texts):
    '''
    Apply basic regex cleaning to a list of texts - used before sentencization
    '''
    return [clean_text(text) for text in texts] # use utils function


def sentencize(texts, n_process=-1, batch_size=100):
    '''
    Use spacy nlp model to sentencize texts with efficient pipe method
    '''
    texts_as_sents = []
    for doc in tqdm(NLP.pipe(texts, n_process=n_process, batch_size=batch_size)):
        sents = np.array([str(sent) for sent in doc.sents]) # get sents as strings from doc object
        texts_as_sents.append([sent for sent in sents if len(sent) >= 2]) # filter out empty or too-short sentences ### this differs from the old version
    return texts_as_sents


def clean_embed(texts, long_texts=False, return_clean_texts=False, sentence_means=True, return_sents=True, show_progress_bar=False, batch_size=32):
    '''
    Input: list of N strings
    Output: Nxd array of embeddings (where d is dimension of embedding space for specified model)
    Optionally sentencize each text, embed each sentence separately, and recombine them with mean averaging
    Optionally return cleaned texts - this is not usually necessary because cleaning steps are minimal
    '''
    print('basic cleaning')
    texts = clean(texts)
    
    # default: texts are not long, i.e. do not require sentencization
    if not long_texts:
        print('embedding')
        embeddings = EMBEDDER.encode(texts, show_progress_bar=show_progress_bar, batch_size=batch_size) # use defined embedder model to embed texts
        
        if return_clean_texts:
            return embeddings, texts
        else:
            return embeddings
    
    # if texts are long, do sentencization
    else:
        print('sentencizing')
        sents = sentencize(texts) # sentencize each element of texts
        flat_sents, inds = flatten(sents) # flatten texts, keep track of original position of each new element with inds
        
        print('embedding')
        embeddings = EMBEDDER.encode(flat_sents, show_progress_bar=show_progress_bar, batch_size=batch_size) # use defined embedder model to embed texts
        embeddings = unflatten_array(embeddings, inds) # re-group sentence embeddings of each original text
        
        if sentence_means:
            embeddings = np.array([np.mean(x, axis=0) for x in embeddings]) # take mean of sentence embeddings of each text
        
        # create results list
        # always contains embeddings, optionally sentences and cleaned texts
        res = [embeddings]
        if return_sents: 
            res.append(sents)
        if return_clean_texts:
            res.append(texts)
        return res