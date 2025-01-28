## This script calculates competitor similarities for each of the coded reference pairs of press releases and texts.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import re

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

from pathlib import Path
parent_folder = str(Path(__file__).parents[1])

## Helper functions.
def jaccard_sim_vec(vec1,vec2):
	""" Use the output from sklearn CountVectorizer to calculate Jaccard similarity efficiently. """
	
	## The size of the intersection of the tokens is the dot product of their count vectors after capping the vecs at 1.
	v1 = np.clip(vec1.todense(),a_min=None,a_max=1)
	try:
		v2 = np.clip(vec2.todense(),a_min=None,a_max=1)
	except AttributeError:  ## This catch is needed for the average vector case.
		v2 = np.clip(vec2,a_min=None,a_max=1)
	
	if np.sum(v1+v2) == 0:
		print('All text is stopwords')
		return np.nan
	else:
		return np.dot(v1,v2.transpose())[0,0]/np.sum(np.clip(v1+v2,a_min=None,a_max=1))

stopwords = list(set([porter_stemmer.stem(w) for w in text.ENGLISH_STOP_WORDS]))
def stemming_tokenizer(str_input,stopwords=stopwords):  ## Source https://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/
	""" Return a tokenized word list after applying the Porter stemmer. """
	words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
	
	words = [porter_stemmer.stem(word) for word in words]
	return words

## Read in the pair data.
df_pairs = pd.read_json(f'{parent_folder}/data/similarity_manual_sample.json')  ## This file includes the labels from the coders.

## Load the press release texts.
pr_texts = {}  ## label: body text
pr_sents = {}  ## label: list of sentences

## Load the press release embedding data to include the raw text.
df_pr_embed = pd.read_json('') 
df_pr_embed['fname'] = df_pr_embed.index  ## Needed to match the datasets.

for pr in sorted(df_pairs.pr_title):
	if not pr_texts.get(pr):
		pr_texts[pr] = df_pr_embed[df_pr_embed.fname==pr+'.txt'].iloc[0].text
		pr_sents[pr] = df_pr_embed[df_pr_embed.fname==pr+'.txt'].iloc[0].sents

## Now add the press release texts to the pair dataframe.
df_pairs['pr_text'] = [pr_texts[p] for p in df_pairs.pr_title]
df_pairs['pr_sents'] = [pr_sents[p] for p in df_pairs.pr_title]

## Calculate the TF-IDF similarities.
## Repeat for sentences.
corpus = [t.lower() for t in df_pairs.text] + [s.lower() for t in df_pairs.pr_sents for s in t]
row_label = len(df_pairs)
corpus_labels = {}  ## tweet row: list of corresponding sentence rows.
for i in range(len(df_pairs)):
	corpus_labels[i] = [row_label + j for j in range(len(df_pairs.pr_sents.iloc[i]))]
	row_label += len(df_pairs.pr_sents.iloc[i]) ## Shift by the number of sentence rows added.
vectorizer = TfidfVectorizer(stop_words=stopwords,tokenizer=stemming_tokenizer)#
X = vectorizer.fit_transform(corpus)

tfidf_sims_avg_vec = []
for i in range(len(df_pairs)):
	tfidf_sims_avg_vec.append(cosine_similarity(X[i],np.mean([X[j].todense() for j in corpus_labels[i]],axis=0))[0][0])
	
df_pairs['sent_avg_vec_tfidf_sim'] = tfidf_sims_avg_vec

## Use the TF-IDF vectors to calcualte the LSA similarities.
## All we need to do is apply SVD to the TF-IDF array. Adapted from https://carpentries-incubator.github.io/python-text-analysis/06-lsa/index.html
from sklearn.decomposition import TruncatedSVD
maxDimensions = min(X.shape)-1
svdmodel = TruncatedSVD(n_components=maxDimensions)
lsa = svdmodel.fit_transform(X)

for k in [9]:
	lsa_trim = lsa[:,:k*100]

	lsa_sims_avg_vec = []
	for i in range(len(df_pairs)):
		lsa_sims_avg_vec.append(cosine_similarity(lsa_trim[i].reshape(1,-1),np.mean([lsa_trim[j].reshape(1,-1) for j in corpus_labels[i]],axis=0))[0][0])
		
	df_pairs[f'sent_avg_vec_lsa_{k*100}_sim'] = lsa_sims_avg_vec

## Calculate the Jaccard keyword similarity.
sent_avg_vec_jaccard_sims = []

## Repeat for the sentences.
corpus = [t.lower() for t in df_pairs.text] + [s.lower() for t in df_pairs.pr_sents for s in t]
vectorizer = CountVectorizer(stop_words=stopwords,tokenizer=stemming_tokenizer)#
X = vectorizer.fit_transform(corpus)

for i in range(len(df_pairs)):
	count_vecs = []
	for j in corpus_labels[i]:
	## The pairs to compare are i and i+len(df_pairs)
		count_vecs.append(X[j].todense())

	sent_avg_vec_jaccard_sims.append(jaccard_sim_vec(X[i],np.where(np.mean(count_vecs,axis=0)>0,1,0)))

df_pairs['sent_avg_vec_jaccard_sim'] = sent_avg_vec_jaccard_sims

## Write the final df to file.
df_pairs.to_json(f'{parent_folder}/data/similarity_metric_comparisons.json')
