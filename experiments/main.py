
from gensim.models import Word2Vec
# all tests conducted on the Wikipedia Corpus
from gensim.corpora import WikiCorpus




# given a text corpus generate word embeddings using word2vec 
def gen_embeddings(path_to_corpus):
	
	sentences = 

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
	return "memes"


# given a text corpus generate word embeddings using word2vec 
def run_tests(embeddings):
	return "memes"
