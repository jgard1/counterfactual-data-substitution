from gensim.models import Word2Vec

# given a directory path, iterates through all json files in directory and loads in the words
# to build up sentences. Basically will set stuff up for running word2vec
def prep_data(dir_path):
	
	all_f_paths = [f.path for f in os.scandir(input_dir) if\
	 (f.is_file() and re.search("json$", f.path))]

	sentences = []
	for f_path in all_f_paths:
	 	with open(f_path, 'r') as file:
 	    	document = json.load(file)
 	  	for sentence in doc_arr:
 	  		as_arr = np.asarray(sentence)
 	  		sentences.append(as_arr[:, 0].tolist())
 	return sentences 
 	    	

# get the data
sentences = rep_data("./modified_wikidata")

# train model
model = Word2Vec(sentences, min_count=1)

print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)


# # summarize the loaded model
# print(model)
# # summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# access vector for one word