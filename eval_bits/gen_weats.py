import json 
import random

with open("names_pairs_1000_scaled.json", 'r') as file:
	names = json.loads(file.read())

with open("cda_default_pairs.json", 'r') as file:
	gendered_phrases = json.loads(file.read())

print("names:"+str(names))
print("\n\n\n\n\n\n\n")
print("gendered_phrases:"+str(gendered_phrases))
all_gendered_words =  gendered_phrases
print("all_gendered_words:"+str(all_gendered_words))

masculine = [masculine for masculine, feminine in all_gendered_words]
feminine = [feminine for masculine, feminine in all_gendered_words]


career= ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
science = ['science','technology','physics','chemistry','einstein','nasa','experiment','astronomy']

family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage','wedding', 'relatives']
art = ['poetry','art','shakespeare','dance','literature','novel','symphony','drama']

occupation_result = {}
occupation_result['target_x'] = masculine 
occupation_result['target_y'] = feminine 
occupation_result['attribute_a'] = career
occupation_result['attribute_b'] = family
occupation_result['section'] = "career_and_family"

academic_result = {}
academic_result['target_x'] = masculine 
academic_result['target_y'] = feminine 
academic_result['attribute_a'] = science
academic_result['attribute_b'] = art
academic_result['section'] = "sceince_and_art"


results = [occupation_result, academic_result]



with open('sections.json', 'w') as f:
    json.dump(results, f)
