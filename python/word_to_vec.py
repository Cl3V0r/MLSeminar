import gensim

model = gensim.models.Word2Vec.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin.gz', binary=True)
model.init_sims(replace=True)

print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1),
    model.doesnt_match("breakfast cereal dinner lunch".split()),
    model.similarity('woman', 'man'))
