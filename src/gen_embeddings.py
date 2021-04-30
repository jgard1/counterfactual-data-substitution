from gensim.models import Word2Vec
import os
import re
import json
import numpy as np

dir_path = "./modified_memes/"
model_name = "memes"
# given a directory path, iterates through all json files in directory and loads in the words
# to build up sentences. Basically will set stuff up for running word2vec
# def prep_data(dir_path):
	
# 	all_f_paths = [f.path for f in os.scandir(dir_path) if\
# 		(f.is_file() and re.search("json$", f.path))]

# 	sentences = []
# 	for f_path in all_f_paths:
# 		print("f_path: "+str(f_path))
# 		with open(f_path, 'r') as file:
# 			document = json.load(file)
# 		for sentence in document:
# 			# print("sentence:"+str(sentence))
# 			as_arr = np.asarray(sentence)
# 			if(as_arr.size != 0):
# 				sentences.append(as_arr[:, 0].tolist())
# 	return sentences 

class Sentences(object):

	def __init__(self, dir_path):
		self.dir_path = dir_path

	def __iter__(self):
		all_f_paths = [f.path for f in os.scandir(self.dir_path) if\
			(f.is_file() and re.search("json$", f.path))]

		sentences = []
		for f_path in all_f_paths:
			print("f_path: "+str(f_path))
			with open(f_path, 'r') as file:
				document = json.load(file)
			for sentence in document:
				# print("sentence:"+str(sentence))
				as_arr = np.asarray(sentence)
				if(as_arr.size != 0):
					sent = as_arr[:, 0].tolist()
					# print("sent:"+str(sent))
					yield sent


# get the data
# sentences = Sentences("./modified_wikicorpus/")
sentences = Sentences(dir_path)

# train model
model = Word2Vec(sentences, min_count=1, vector_size = 300)
word_vectors = model.wv
word_vectors.save(str(model_name)+".wordvectors")
# summarize the loaded model
# print(model)
# summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# access vector for one word


# save model
model.save(str(model_name)+'.bin')
# load model
# new_model = Word2Vec.load('dank_model.bin')
# print(new_model)


