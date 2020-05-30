# -*- coding: utf-8 -*-
# from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from twenty_newsgroups import load_20newsgroups
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Load some categories from the training set
dataset = load_20newsgroups(data_home='./', subset='all', categories=None)
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(dataset.data)

# perform clustering
k = 1
km = KMeans(n_clusters=k, max_iter=100, n_init=1)
km.fit_transform(X)

centroids = km.cluster_centers_  # it is a matrix whose size is k by feature_num
terms = vectorizer.get_feature_names()
for i in range(k):
    print('Cluster %d:' % i)
    word2tfidf = {k: v for (k, v) in zip(terms, np.ravel(centroids[i]))}
    print("word2tfidf:", word2tfidf)
    # wordcloud = WordCloud().generate_from_frequencies(word2tfidf)
    # # Output the generated file to current folder
    # wordcloud.to_file('cluster_{}.png'.format(i))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

