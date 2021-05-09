from gensim.models import Word2Vec
import os
import re
import json
import numpy as np
import argparse

# need this class in order to iterate over the sentecnes and not crash the model 
class Sentences(object):

	def __init__(self, dir_path):
		self.dir_path = dir_path

	def __iter__(self):
		all_f_paths = [f.path for f in os.scandir(self.dir_path) if\
			(f.is_file() and re.search("json$", f.path))]

		for f_path in all_f_paths:
			# print("f_path: "+str(f_path))
			with open(f_path, 'r') as file:
				content = file.read()
				# print("content[63385740:63385756]"+str(content[63385740:63385756]))
				document = json.loads(content)
			for sentence in document:
				# print("sentence:"+str(sentence))
				as_arr = np.asarray(sentence)
				if(as_arr.size != 0):
					sent = as_arr[:, 0].tolist()
					print("sent:"+str(sent))
					yield sent




parser = argparse.ArgumentParser()
parser.add_argument("dir_path", help="The path to the directory containing all the sentences for trainign the model")
parser.add_argument("model_path", help="the full path you want the model outputted to")
args = parser.parse_args()

print("generating embeddings for directory:"+str(args.dir_path))
sentences = Sentences(args.dir_path)
model = Word2Vec(sentences, min_count=10, workers=8, vector_size=100)
word_vectors = model.wv

word_vectors.save(str(args.model_path)+".wordvectors")
# model.save(str(args.model_path)+'.bin')



