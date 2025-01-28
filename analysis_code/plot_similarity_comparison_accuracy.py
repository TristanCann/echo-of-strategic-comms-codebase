## This script reproduces the similarity threshold accuracy figure from the SI using the differnt metrics for comparison.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
from pathlib import Path
parent_folder = str(Path(__file__).parents[1])

## Parameters
min_x = 0
max_x = 1

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=14)
arr_blue, arr_green, arr_red, arr_purple = np.array([[0.368, 0.506, 0.709]]), np.array([[0.560, 0.691, 0.194]]), np.array([[0.922, 0.385, 0.209]]), np.array([[0.528, 0.470, 0.701]])
blue, green, red, purple, white  = (0.368, 0.506, 0.709), (0.560, 0.691, 0.194), (0.922, 0.385, 0.209), (0.528, 0.470, 0.701), (1,1,1)

## Read in the data.
df = pd.read_json(f'{parent_folder}/data/similarity_metric_comparisons.json')

## For each threshold value.
emb_f1 = []
emb_b_acc = []

tfidf_f1 = []
tfidf_b_acc = []

lsa_f1 = {i:[] for i in [9]}
lsa_b_acc = {i:[] for i in [9]}

jaccard_f1 = []
jaccard_b_acc = []
for v in np.linspace(min_x,max_x,num=300):
	thresh_labels = [0 if i < v else 1 for i in df.sim]

	## Calculate the accuracy figures.
	## Accuracy
	emb_b_acc.append(sklearn.metrics.balanced_accuracy_score(df.ground_truth,thresh_labels))
	
	## F1
	emb_f1.append(sklearn.metrics.f1_score(df.ground_truth,thresh_labels))
	
	## Repeat for TF-IDF.
	thresh_labels = [0 if i < v else 1 for i in df.sent_avg_vec_tfidf_sim]

	## Accuracy
	tfidf_b_acc.append(sklearn.metrics.balanced_accuracy_score(df.ground_truth,thresh_labels))
	
	## F1
	tfidf_f1.append(sklearn.metrics.f1_score(df.ground_truth,thresh_labels))
	
	## Repeat for LSA.
	for i in [9]:
		thresh_labels = [0 if i < v else 1 for i in df[f'sent_avg_vec_lsa_{(i)*100}_sim']]

		## Accuracy
		lsa_b_acc[i].append(sklearn.metrics.balanced_accuracy_score(df.ground_truth,thresh_labels))
		
		## F1
		lsa_f1[i].append(sklearn.metrics.f1_score(df.ground_truth,thresh_labels))
	
	## Repeat for Jaccard
	thresh_labels = [0 if i < v else 1 for i in df.sent_avg_vec_jaccard_sim]

	## Accuracy
	jaccard_b_acc.append(sklearn.metrics.balanced_accuracy_score(df.ground_truth,thresh_labels))
	
	## F1
	jaccard_f1.append(sklearn.metrics.f1_score(df.ground_truth,thresh_labels))

## Plot the accuracy figures.
plt.figure(figsize=(7.5,4.5))
plt.hlines(xmin=min_x,xmax=max_x,y=0,color='darkgrey',alpha=0.7)
plt.plot(np.linspace(min_x,max_x,num=300),emb_b_acc,label='Embedding',color=blue)
plt.plot(np.linspace(min_x,max_x,num=300),tfidf_b_acc,label='TF-IDF',color=red,ls='--')
plt.plot(np.linspace(min_x,max_x,num=300),jaccard_b_acc,label='Jaccard',color=green,ls=':')
for i in [9]:
	plt.plot(np.linspace(min_x,max_x,num=300),lsa_b_acc[i],label=f'LSA',color='gold',ls='-.')

## Annotate the optimal threshold for each metric.
ax = plt.gca()
ax.vlines(ymin=0,ymax=emb_b_acc[np.argmax(emb_b_acc)],x=np.linspace(min_x,max_x,num=300)[np.argmax(emb_b_acc)],color=blue,alpha=0.2)
ax.vlines(ymin=0,ymax=tfidf_b_acc[np.argmax(tfidf_b_acc)],x=np.linspace(min_x,max_x,num=300)[np.argmax(tfidf_b_acc)],color=red,alpha=0.2, ls='--')
ax.vlines(ymin=0,ymax=jaccard_b_acc[np.argmax(jaccard_b_acc)],x=np.linspace(min_x,max_x,num=300)[np.argmax(jaccard_b_acc)],color=green,alpha=0.2,ls=':')
ax.vlines(ymin=0,ymax=lsa_b_acc[9][np.argmax(lsa_b_acc[9])],x=np.linspace(min_x,max_x,num=300)[np.argmax(lsa_b_acc[9])],color='gold',alpha=0.2,ls='-.')

plt.xlim(min_x,max_x)
plt.ylim(bottom=0)
plt.xlabel('Similarity threshold')
plt.ylabel('Balanced accuracy')
plt.legend(frameon=False,loc='lower center')
plt.savefig('figs/Fig2a.pdf',bbox_inches='tight')
plt.close()

plt.figure(figsize=(7.5,4.5))
plt.hlines(xmin=min_x,xmax=max_x,y=0,color='darkgrey',alpha=0.7)
plt.plot(np.linspace(min_x,max_x,num=300),emb_f1,label='Embedding',color=blue)
plt.plot(np.linspace(min_x,max_x,num=300),tfidf_f1,label='TF-IDF',color=red,ls='--')
plt.plot(np.linspace(min_x,max_x,num=300),jaccard_f1,label='Jaccard',color=green,ls=':')
for i in [9]:
	plt.plot(np.linspace(min_x,max_x,num=300),lsa_f1[i],label=f'LSA{i*100}',color='gold',ls='-.')

ax = plt.gca()
ax.vlines(ymin=0,ymax=emb_f1[np.argmax(emb_f1)],x=np.linspace(min_x,max_x,num=300)[np.argmax(emb_f1)],color=blue,alpha=0.2)
ax.vlines(ymin=0,ymax=tfidf_f1[np.argmax(tfidf_f1)],x=np.linspace(min_x,max_x,num=300)[np.argmax(tfidf_f1)],color=red,alpha=0.2, ls='--')
ax.vlines(ymin=0,ymax=jaccard_f1[np.argmax(jaccard_f1)],x=np.linspace(min_x,max_x,num=300)[np.argmax(jaccard_f1)],color=green,alpha=0.2,ls=':')
ax.vlines(ymin=0,ymax=lsa_f1[9][np.argmax(lsa_f1[9])],x=np.linspace(min_x,max_x,num=300)[np.argmax(lsa_f1[9])],color='gold',alpha=0.2,ls='-.')

plt.xlim(min_x,max_x)
plt.ylim(bottom=0)
plt.xlabel('Similarity threshold')
plt.ylabel('F1')
plt.savefig('figs/Fig2b.pdf',bbox_inches='tight')
plt.close()

## Print out the maximum balanced accuracy scores we observed in this search, convert to a threshold.
xvals = np.linspace(min_x,max_x,num=300)
print(f'Embedding bal. acc. maximises at {emb_b_acc[np.argmax(emb_b_acc)]} with threshold {xvals[np.argmax(emb_b_acc)]}')
print(f'TFIDF bal. acc. maximises at {tfidf_b_acc[np.argmax(tfidf_b_acc)]} with threshold {xvals[np.argmax(tfidf_b_acc)]}')
print(f'Jaccard bal. acc. maximises at {jaccard_b_acc[np.argmax(jaccard_b_acc)]} with threshold {xvals[np.argmax(jaccard_b_acc)]}')
print(f'LSA bal. acc. maximises at {lsa_b_acc[9][np.argmax(lsa_b_acc[9])]} with threshold {xvals[np.argmax(lsa_b_acc[9])]}')

print()

print(f'Embedding f1 maximises at {emb_f1[np.argmax(emb_f1)]} with threshold {xvals[np.argmax(emb_f1)]}')
print(f'TFIDF f1 maximises at {tfidf_f1[np.argmax(tfidf_f1)]} with threshold {xvals[np.argmax(tfidf_f1)]}')
print(f'Jaccard f1 maximises at {jaccard_f1[np.argmax(jaccard_f1)]} with threshold {xvals[np.argmax(jaccard_f1)]}')
print(f'LSA f1 maximises at {lsa_f1[9][np.argmax(lsa_f1[9])]} with threshold {xvals[np.argmax(lsa_f1[9])]}')

## Classify the similarities based on the preferred thresholds.
df['emb_label'] = df.sim>0.7
df.emb_label = df.emb_label.astype('int')
df['tfidf_label'] = df.sent_avg_vec_tfidf_sim>0.13
df.tfidf_label = df.tfidf_label.astype('int')
df['jaccard_label'] = df.sent_avg_vec_jaccard_sim>0.03
df.jaccard_label = df.jaccard_label.astype('int')
df['lsa_label'] = df.sent_avg_vec_lsa_900_sim > 0.20
df['lsa_label'] = df.lsa_label.astype('int')

## Calculate precision, recall and balanaced accuracy of each metric.
precision_emb = sklearn.metrics.precision_score(df.ground_truth, df.emb_label)
precision_tfidf = sklearn.metrics.precision_score(df.ground_truth, df.tfidf_label)
precision_jaccard = sklearn.metrics.precision_score(df.ground_truth, df.jaccard_label)
precision_lsa = sklearn.metrics.precision_score(df.ground_truth, df.lsa_label)

recall_emb = sklearn.metrics.recall_score(df.ground_truth, df.emb_label)
recall_tfidf = sklearn.metrics.recall_score(df.ground_truth, df.tfidf_label)
recall_jaccard = sklearn.metrics.recall_score(df.ground_truth, df.jaccard_label)
recall_lsa = sklearn.metrics.recall_score(df.ground_truth, df.lsa_label)

b_acc_emb = sklearn.metrics.balanced_accuracy_score(df.ground_truth, df.emb_label)
b_acc_tfidf = sklearn.metrics.balanced_accuracy_score(df.ground_truth, df.tfidf_label)
b_acc_jaccard = sklearn.metrics.balanced_accuracy_score(df.ground_truth, df.jaccard_label)
b_acc_lsa = sklearn.metrics.balanced_accuracy_score(df.ground_truth, df.lsa_label)

## Print a table for ease of copying.
print( '#######################################################')
print( '#           # Embedding #  TFIDF  # Jaccard #   LSA   #')
print( '#######################################################')
print(f'# Precision #  {precision_emb:6.5f}  # {precision_tfidf:6.5f} # {precision_jaccard:6.5f} # {precision_lsa:6.5f} #')
print(f'# Recall    #  {recall_emb:6.5f}  # {recall_tfidf:6.5f} # {recall_jaccard:6.5f} # {recall_lsa:6.5f} #')
print(f'# Bal. Acc. #  {b_acc_emb:6.5f}  # {b_acc_tfidf:6.5f} # {b_acc_jaccard:6.5f} # {b_acc_lsa:6.5f} #')
print( '#######################################################')