import sys
from substitutor import Substitutor
from utils import load_json_pairs
import os
import json
import argparse
import random


def process_text_files(input_dir, output_dir, substitutor, invert_cond):

	all_f_paths = [f.path for f in os.scandir(input_dir) if f.is_file()]
	f = 1

	for f_path in all_f_paths:
		if f % 5 == 0:
			print("progress: " + str((f / len(all_f_paths)) * 100) + "%")
		with open(f_path, 'rb') as file:
			text = str(file.read())
		sentences = format_tagged_data(text)

		if invert_cond == "original_text":
			write_file(sentences, f_path, output_dir)

		# have to perform on file level for control group
		if invert_cond == "invert_control":
			if bool(random.getrandbits(1)):  # invert file 50% of the time
				print("invert sentences")
				sentences = invert_file(sentences, substitutor)
			else:
				print("don't invert sentences")

			write_file(sentences, f_path, output_dir)
		else:
			sentences = invert_file(sentences, substitutor)
			write_file(sentences, f_path, output_dir)
		f += 1


def invert_file(sentences, substitutor):
	flipped_doc = []

	for sentence in sentences:
		flipped_sentence = substitutor.invert_document(sentence)
		flipped_doc.append(flipped_sentence)
	return flipped_doc


def write_file(flipped_doc, f_path, output_dir):
	print("Starting file: " + f_path)
	text = ""

	f_hierarchy = f_path.split("/")
	f_name = f_hierarchy[len(f_hierarchy) - 1]
	no_exetension = (f_name.split("."))[0]

	new_f_path = output_dir +str(no_exetension) + "_modified.json"
	print(new_f_path)
	with open(new_f_path, 'w') as file:
		json.dump(flipped_doc, file)


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

substitutor = Substitutor(base_pairs, invert_cond=args.experiment_condition, name_pairs=name_pairs)


print("starting to process text files in dir:"+str(args.in_dir))
print("\n\n\n\n\n\n")
process_text_files(args.in_dir, args.out_dir, substitutor, args.experiment_condition)
print("done processing text files")



