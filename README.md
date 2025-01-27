# Codebase and data references for "Using Semantic Similarity to Measure the Echo of Strategic Communications"

We developed a novel method for measuring the 'echo' of strategic communications that uses semantic similarity to identify texts that repeat similar messages without requiring keyword searches. This method was tested on a set of press releases from environmentally-active organisations and their potential impact on Twitter discussion of climate change. Full details of this work are available at [paper under review].

This codebase has been tested on **Python 3.12.3** using the module versions listed in **requirements.txt**. Much of the methodology here is compatible with other Python and module versions, but note that the sentencisation behaviour of Spacy has been observed to change and will therefore have a small effect on similarity scores for longer texts.

The modules used in this library can be installed using:
`pip install -r requirements.txt`

The required embedding model can be downloaded using:
`python -m spacy download en_core_web_sm`

## Data
 - **pr_urls.txt**: List of URLs for all press releases included in this study, one per line. 
 - **tweet_ids.txt**: List of IDs for all tweets included in this study, one per line. This file is too large to be stored in Github and can be found on [Figshare](https://dx.doi.org/10.6084/m9.figshare.28196000).

## echo_code
 ### 1_embedding
 This folder contains the necessary code to compute the text embeddings for the press releases and tweets.
    - **embed_prs.py**: Calculate the text embeddings for the press releases, including sentencisation and averaging to determine document-level embeddings.
    - **embed_text.py**: General functions for cleaning and embedding any text documents.
    - **embed_tweets.py**: Calculate the text embeddings for the tweets.

 ### 2_similarities
 This folder contains the necessary code to calculate the cosine similarity between the press release and tweet embeddings.
    - **text_tweet_sims.py**: Use the previously computed embeddings to calculate the cosine similarity between a press release and all tweets collected within a few days of the press release publication.

 ### 3_echo
 This folder contains the necessary code to use the computed cosine similarity values to calculate the echo of a given press release.
    - **calculate_pr_echo.py**: Use the previously computed similarity scores to calculate the both echo metrics of a range of (pre,post) window sizes.
   
 **utils.py**
 This file contains several helper functions for file I/O, text cleaning and calculation of the echo score on a set of similarities.

## analysis_code

 