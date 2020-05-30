from gensim.models import Word2Vec

word_vectors = Word2Vec.load('./20news-vectors-negative100.model')

# reference https://radimrehurek.com/gensim/models/keyedvectors.html
result = word_vectors.wv.most_similar(positive=['woman', 'husband'], negative=['man'])
print(result)
similarity = word_vectors.wv.similarity('football', 'baseball')
print(similarity)
similarity = word_vectors.wv.similarity('football', 'mac')
print(similarity)
result = word_vectors.wv.similar_by_word('baseball')
print(result)
