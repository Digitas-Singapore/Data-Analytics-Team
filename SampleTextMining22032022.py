# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Chilin's Text Analytics practice : Text preparation step by step
"""

import pandas  # Package as analysis engine of relational and labeled data with fast, flexible and expressive data structures https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html
import nltk
from nltk import word_tokenize, FreqDist, pos_tag, RegexpParser
import os
import json
import string
from nltk import word_tokenize, FreqDist, pos_tag
from nltk.corpus import stopwords

# load required data for NLTK 
# Note: on local machine, you just need to run it once only using "nltk.download()"
#       at the 'NLTK Downloader' dialogue window, choose 'book', then click 'Download'

nltk.download('punkt')  #sentence tokenizer
nltk.download('stopwords')  #removes common words with little or no meaning
nltk.download('wordnet')  #word database of English nouns, adjectives, adverts and verbs
nltk.download('averaged_perceptron_tagger')  #Part of speech tagger
nltk.download('words') #loading corpora of English words

# Fetch a single <10MB file using the raw GitHub URL.
!curl --remote-name \
-H 'Accept: application/vnd.github.v3.raw' \
--location https://raw.githubusercontent.com/Digitas-Singapore/Chilin-s-Github-datasets/main/osha.txt

# Open and read in data from osha.txt
data_df  =  pandas.read_csv("osha.txt",delimiter="\t", header=None) #reads text data into data frame

data_title = data_df .iloc[:, 1]  #iloc:  indexing based on integer position - titles from frame 1
data_text = data_df .iloc[:, 2]  #iloc:  indexing based on integer position - text body from frame 2

wnl = nltk.WordNetLemmatizer()  
# Grouping inflected form of a word for analysis.  Unlike stemming, it analyses context of the word.  
# Good example being The word "better" has "good" as its lemma. This link is missed by stemming, as it requires a dictionary look-up.

# Getting POS info
pos_text = [ pos_tag(word_tokenize(f)) for f in data_text ]
pos_title = [ pos_tag(word_tokenize(f)) for f in data_title ]

#Construct lemmatized forms of plural words - in lower case as prerequisite for lemmatization
def lemmaN(wpos):
  nouns = [ (w, pos) for (w, pos) in wpos if pos in ('NN', 'NNS')]  #Assign words to "nouns" string if defined as singular or plural nouns
  # NN	noun, singular 'desk'
  # NNS	noun plural	'desks'
  lemman = [ wnl.lemmatize(w.lower(), pos = 'n') if pos == 'NNS' else w.lower() for (w, pos) in nouns ]
  return lemman

nouns = [ lemmaN(pos_tag(word_tokenize(f))) for f in data_text ]
# tokenize and lemmatize words indexed in data_text (from data_df)

nounsets = [ list(set(f)) for f in nouns]  #creates noun list

from google.colab import drive
drive.mount('/content/drive')

# View snapshot of records from data frame (title frame 1 and body text 2)

data_df

#Quality control check for contents in title list and body text lists
print(data_title[4]) #Display snapshot of title list
print(data_text[4]) #Display snapshot of body text list

nouns = [ lemmaN(f) for f in pos_text ]     #Assigns lemmatized nouns from NN and NNS in body text 
nounsets = [ list(set(f)) for f in nouns ]  #Assigns lemmatized nouns into list nounsets
noun_flat = [ c for l in nounsets for c in l ]

# Counting nouns by document frequency
fd_n = FreqDist(noun_flat)
fd_n.most_common(50)

# Finding number of unique words relating to worker
from nltk.corpus import wordnet as wn
ssW = wn.synsets('worker')[0]
#ssW = wn.synsets('worker')[0]  Assigns worker related words to wn synset
hypsW = list(set([w for s in ssW.closure(lambda s:s.hyponyms()) 
                        for w in s.lemma_names()]))
hypsW.sort()
len(hypsW)
#ssE = wn.synsets('employee')[0]
#hypsW = list(set([w for s in ssE.closure(lambda s:s.hyponyms())
#                       for w in s.lemma_names()]))
#hypsEW = list(set(hypsE + hypsW))

# Search results of occupations listed in osha accident report, using words from worker wordnet
results_occ = [ (w, fd_n[w]) for w in fd_n.keys() if w in hypsW ]
sorted_occ = sorted(results_occ, key=lambda x: x[1], reverse=True)
sorted_occ[:50]

# Finds number of unique words relating to body parts
from nltk.corpus import wordnet as wn
ssB = wn.synsets('body_part')[0]
print(ssB)
ssB.hypernyms()
#ssB = wn.synsets('body')[0]  Assigns body part related words to wn synset
hypsB = list(set([w for s in ssB.closure(lambda s:s.hyponyms()) 
                        for w in s.lemma_names()]))
hypsB.sort()

len(hypsB)

# Returns unique words relating to body parts
results_bp = [ (w, fd_n[w]) for w in fd_n.keys() if w in hypsB ]
sorted_bp = sorted(results_bp, key=lambda x: x[1], reverse=True)
sorted_bp[:50]

# A little exploration: What are the unique nouns are there in the osha article set, for body parts? 
sorted_bp[:30]

#How many unique words for body parts?
uniqueB = [ c for l in sorted_bp for c in l ]
print("No. of unique nouns on body parts: ", len(uniqueB))
#  Writing results to txt file
with open('risky_occupations.txt', 'w') as filehandle:
    filehandle.write('%s\n' % uniqueB)

# Frequency distribution and plot of unique words for body parts

fdB = nltk.FreqDist(uniqueB)
print(fdB.most_common(100))
fdB.plot(100)

# A little exploration: What are the unique nouns are there in the osha article set, for occupations? 
sorted_occ[:30]

#How many unique words for occupations?
uniqueOcc = [ c for l in sorted_occ for c in l ]
print("No. of unique nouns: ", len(uniqueOcc))
#Writing results to text file
with open('risky_occupations.txt', 'w') as filehandle:
    filehandle.write('%s\n' % uniqueOcc)

# Frequency distribution and plot of unique words associated with risky occupations
fdOcc = nltk.FreqDist(uniqueOcc)
print(fdOcc.most_common(100))
fdOcc.plot(100)

#Graphing using a switched axis
mostcommon = fdOcc.most_common(100)
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

# What about bigrams and trigrams for unique occupations??
bigrOcc = nltk.bigrams(uniqueOcc[:10])
trigrOcc = nltk.trigrams(uniqueOcc[:10])
uniqueOcc[:10]

list(bigrOcc)

list(trigrOcc)

# What about bigrams and trigrams for unique body parts?
bigrB = nltk.bigrams(uniqueB[:10])
trigrB = nltk.trigrams(uniqueB[:10])
uniqueB[:10]

list(bigrB)

list(trigrB)

# Colab environment already has wordcloud package
#!pip install wordcloud

#if you have not installed matplotlib, pandas, please also install them
# import wordcloud
# from wordcloud import WordCloud, ImageColorGenerator
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
