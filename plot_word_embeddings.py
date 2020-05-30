from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from matplotlib import pylab
import numpy as np


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


num_points = 400
word2vec = KeyedVectors.load_word2vec_format('./20news-vectors-negative100.bin', binary=True)
words = word2vec.index2entity[:num_points]

vectors = []
for w in words:
    vectors.append(word2vec[w])
vectors = np.asarray(vectors, dtype=np.float32)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(vectors)
plot(two_d_embeddings, words)
