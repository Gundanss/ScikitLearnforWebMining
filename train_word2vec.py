from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize

newsgroups_data = fetch_20newsgroups(subset='all')
sentences = []
for doc in newsgroups_data.data:
    for sent in sent_tokenize(doc):
        word_list = word_tokenize(sent)
        sentences.append(word_list)

# reference https://radimrehurek.com/gensim/models/word2vec.html
print('Start training!')
model = Word2Vec(sentences, sg=1, hs=0, size=100, min_count=5, max_vocab_size=50000)  # skip-gram with negative sampling
model.save('20news-vectors-negative100.model')
model.wv.save_word2vec_format('20news-vectors-negative100.bin', binary=True)
print('Done training!')
