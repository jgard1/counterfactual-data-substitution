import sys
from substitutor import Substitutor
from utils import load_json_pairs
import os
import json
import argparse

def process_text_files(input_dir, output_dir, substitutor):
	
	all_f_paths = [f.path for f in os.scandir(input_dir) if f.is_file()]
	f = 0
	for f_path in all_f_paths:
		print("Starting file: " + f_path)
		if f % 10 == 0:
			print("file progress: " + str((f / len(all_f_paths)) * 100) + "%")
		text = ""
		with open(f_path, 'rb') as file:
			text = str(file.read())

		sentences = format_tagged_data(text)
		# print("sentences: "+str(sentences))
		flipped_doc = []
		i = 1
		for sentence in sentences:
			if i % 10000 == 0:
				print("progress: " + str((i / len(sentences)) * 100) + "%")
			flipped_sentence = substitutor.invert_document(sentence)
			# print("flipped sentence: "+str(flipped_sentence))
			# print("\n\n\n")
			flipped_doc.append(flipped_sentence)
			i += 1

		f_hierarchy = f_path.split("/")
		f_name = f_hierarchy[len(f_hierarchy) - 1]
		no_exetension = (f_name.split("."))[0]

		new_f_path = output_dir +str(no_exetension) + "_modified.json"
		print(new_f_path)
		with open(new_f_path, 'a') as file:
			json.dump(flipped_doc, file)
		f += 1



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
				phrase_text = fields[0]
				pos = fields[2]
				for word in phrase_text.split("_"):
					if (word not in punctuation):
						ret_sent.append([word, pos])
		ret_sentences.append(ret_sent)
	return ret_sentences



parser = argparse.ArgumentParser()
parser.add_argument("experiment_condition", help="The experiment condition(e.g. control, CDA, etc.)")
parser.add_argument("in_dir", help="The number of permutations")
parser.add_argument("out_dir", help="The name and location of the target .csv result file")

args = parser.parse_args()

# Unit testing 
base_pairs = load_json_pairs('../data/cda_default_pairs.json')
name_pairs = load_json_pairs('../data/names_pairs_1000_scaled.json')
# Initialise a substitutor with a list of pairs of gendered words (and optionally names)

substitutor = Substitutor(base_pairs, name_pairs=name_pairs)
	# , condition = args.experiment_condition)


print("starting to process text files")
process_text_files(args.in_dir, args.out_dir, substitutor)
print("done processing text files")



