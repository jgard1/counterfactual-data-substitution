import sys
from src.substitutor import Substitutor
from src.utils import load_json_pairs
# Example text which requires NER and POS information to properly invert

base_pairs = load_json_pairs('../data/cda_default_pairs.json')
name_pairs = load_json_pairs('../data/names_pairs_1000_scaled.json')

# Initialise a substitutor with a list of pairs of gendered words (and optionally names)
substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

flipped = substitutor.invert_document(text)

dir_path = "../wikidata/"
all_f_paths = [f.path for f in os.scandir(dir_path) if f.is_file()]
for f_path in all_f_paths:
	
	with open(f_path, 'r') as file:
		text = file.read().replace('\n', '')
	
	flipped = substitutor.invert_document(text)
	
	f_hierarchy = f_path.split("/")
	f_name = f_hierarchy[len(f_hierarchy) - 1]
	no_exetension = (f_name.split("."))[0]
	
	new_f_path = "../modified_wikidata/" +str(f_name) + "_modified.txt"

	with open(new_f_pat, 'w') as file:
		text = file.write(flipped)


# It correctly doesn't flip the sentence ending noun "amber", and properly converts "her" to "his" not "him"

# If you want to apply an intervention probablistically, use the method
flipped = substitutor.probablistic_substitute([text, text, text, text])
# which takes a list and returns a generator which flips 50% of documents

print("50% chance flipped: {}".format(next(flipped)))
print("50% chance flipped: {}".format(next(flipped)))
print("50% chance flipped: {}".format(next(flipped)))
print("50% chance flipped: {}".format(next(flipped)))

