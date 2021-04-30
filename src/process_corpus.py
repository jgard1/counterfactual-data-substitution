import sys
from substitutor import Substitutor
from utils import load_json_pairs
import os
import json

def process_text_files(input_dir, output_dir, substitutor):
	
	all_f_paths = [f.path for f in os.scandir(input_dir) if f.is_file()]
	
	for f_path in all_f_paths:

		text = ""
		with open(f_path, 'rb') as file:
			text = str(file.read())

		sentences = format_tagged_data(text)
		# print("sentences: "+str(sentences))
		flipped_doc = []
		for sentence in sentences:
			flipped_sentence = substitutor.invert_document(sentence)
			# print("flipped sentence: "+str(flipped_sentence))
			# print("\n\n\n")
			flipped_doc.append(flipped_sentence)

		f_hierarchy = f_path.split("/")
		f_name = f_hierarchy[len(f_hierarchy) - 1]
		no_exetension = (f_name.split("."))[0]

		new_f_path = out_dir +str(no_exetension) + "_modified.json"

		with open(new_f_path, 'a') as file:
			json.dump(flipped_doc, file)



# # takes in tagged text data in the format of the wikicorpus dataset and returns the 
# # data in a nice array format with punctuation reemoved. 
# def format_tagged_data(text):
# 	punctuation = {",", ";", ".", "?", "!", "\"", "\'", "\\", ":"}

# 	lines = text.split("\n")
# 	lines = lines[1:] # get rid of first line since it's just metadata
# 	sent_lst = [] # list of lists that stores words and their parts of speech
# 	for line in lines: 
# 		sentence = []
# 		if line != "": # this seems to be a delimeter line
# 			print()
# 			fields = line.split(" ")
# 			phrase = fields[0]
# 			pos = fields[2]
# 			for word in phrase.split("_"):
# 				if (word not in punctuation):
# 					sentence.append([word, pos])

# 		else: # had a blank line 
# 			print("New sentence: "+str(sentence))
# 			sent_lst.append(sentence)
# 	return sent_lst


# takes in tagged text data in the format of the wikicorpus dataset and returns the 
# data in a nice array format with punctuation reemoved. 
def format_tagged_data(text):
	text = str(text)
	punctuation = {",", ";", ".", "?", "!", "\"", "\'", "\\", ":"}
	sentences = text.split("\\n\\n")
	sentences[0] = "\\n".join((sentences[0].split("\\n"))[1:])
	ret_sentences = []
	for sentence in sentences: 
		# print("sentence: "+str(sentence))
		ret_sent = []
		phrases = sentence.split("\\n")
		for phrase in phrases: 
			fields = phrase.split(" ")
			if(len(fields) == 4):
				# print("fields: "+str(fields))
				phrase_text = fields[0]
				pos = fields[2]
				for word in phrase_text.split("_"):
					if (word not in punctuation):
						ret_sent.append([word, pos])
		ret_sentences.append(ret_sent)
	return ret_sentences


# Unit testing 
base_pairs = load_json_pairs('../data/cda_default_pairs.json')
name_pairs = load_json_pairs('../data/names_pairs_1000_scaled.json')
# Initialise a substitutor with a list of pairs of gendered words (and optionally names)
substitutor = Substitutor(base_pairs, name_pairs=name_pairs, spacy_model ="en_core_web_md")

in_dir = "../tagged_wikidata/"
out_dir = "./modified_wikicorpus/"

print("starting to process text files")
process_text_files(in_dir, out_dir, substitutor)
print("done processing text files")



