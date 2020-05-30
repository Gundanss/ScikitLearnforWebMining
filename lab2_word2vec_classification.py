from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors
from nltk import word_tokenize
import numpy as np


def get_doc_embedding(text, word_vectors):
    vector_list = []
    word_list = word_tokenize(text)
    for w in word_list:
        if w in word_vectors:
            # skip the words that are not in word2vec
            vector_list.append(word_vectors[w])

    doc_matrix = np.asarray(vector_list, dtype=np.float32)
    doc_vec = np.mean(doc_matrix, axis=0)

    return doc_vec


def get_data_embedding(doc_list, word_vectors):
    vector_list = []
    for doc in doc_list:
        doc_vec = get_doc_embedding(doc, word_vectors)
        vector_list.append(doc_vec)
    doc_matrix = np.asarray(vector_list, dtype=np.float32)

    return doc_matrix


categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories)
twenty_test = fetch_20newsgroups(subset='test', categories=categories)

# word embeddings
# visit https://drive.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM to download word2vec trained on Google News
# for details see https://code.google.com/archive/p/word2vec/
word2vec = KeyedVectors.load_word2vec_format('./20news-vectors-negative100.bin', binary=True)
X_train = get_data_embedding(twenty_train.data, word2vec)
X_test = get_data_embedding(twenty_test.data, word2vec)

clf = LogisticRegression().fit(X_train, twenty_train.target)
predicted = clf.predict(X_test)

print('Accuracy: %.3f\n' % np.mean(predicted == twenty_test.target))
