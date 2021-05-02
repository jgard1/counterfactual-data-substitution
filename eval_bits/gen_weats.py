import json 
import random
import numpy as np 

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

caucasian_female_names = ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah']
caucasian_male_names = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd']

black_female_names = ['Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latonya', 'Latoya', 'Tamika', 'Tanisha']
black_male_names = ['Darnell', ' Hakim', 'Jamal', 'Jermaine', 'Kareem', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone']

caucasian_stereotypes = ["manager", "executive", "redneck", "hillbilly", "leader", "farmer","suburbs","rural","golf"]
black_steretypes = ["slave", "musician", "runner", "criminal", "homeless","urban","ghetto","rapper","basketball"] 

negative_words=["depressed","disappointed","discouraged","ashamed","powerless","diminished","guilty","dissatisfied","miserable","detestable","repugnant","despicable","disgusting","abominable","terrible","despair","sulky","bad","confused","upset","doubtful","uncertain","indecisive","perplexed","embarrassed","hesitant","shy","stupefied","disillusioned","unbelieving","skeptical","distrustful","misgiving","lost","unsure","uneasy","pessimistic","tense","HELPLESS","incapable","alone","paralyzed","fatigued","useless","inferior","vulnerable","empty","forced","hesitant","despair","frustrated","distressed","woeful","pathetic","tragic","dominated","angry","irritated","enraged","hostile","insulting","annoyed","upset","hateful","offensive","bitter","aggressive","resentful","inflamed","provoked","incensed","infuriated","cross","boiling","fuming","afraid","fearful","terrified","suspicious","anxious","alarmed","panic","nervous","scared","worried","frightened","timid","shaky","restless","doubtful","threatened","cowardly","quaking","wary","hurt","crushed","tormented","deprived","pained","tortured","dejected","rejected","injured","offended","afflicted","aching","victimized","heartbroken","agonized","appalled","humiliated","wronged","alienated","sad","tearful","sorrowful","pained","grief","anguish","desolate","desperate","pessimistic","unhappy","lonely","grieved","mournful","dismayed","indifferent","insensitive","dull","nonchalant","neutral","reserved","weary","bored","preoccupied","cold","disinterested"]
positive_words=["happy","great","joyous","lucky","fortunate","delighted","overjoyed","gleeful","thankful","important","festive","ecstatic","glad","cheerful","elated","jubilant","ALIVE","playful","courageous","energetic","liberated","optimistic","impulsive","free","animated","spirited","thrilled","wonderful","good","calm","peaceful","at ease","comfortable","pleased","encouraged","clever","surprised","content","quiet","certain","relaxed","serene","reassured","open","understanding","confident","reliable","easy","amazed","free","sympathetic","interested","satisfied","receptive","accepting","kind","interested","concerned","affected","fascinated","intrigued","absorbed","inquisitive","engrossed","curious","drawn toward","positive","eager","keen","earnest","intent","inspired","determined","excited","enthusiastic","bold","brave","daring","optimistic","strong","impulsive","free","sure","certain","rebellious","unique","dynamic","tenacious","hardy","secure","confident","challenged","love","loving","considerate","affectionate","sensitive","tender","devoted","attracted","passionate","admiration","warm","touched","close","comforted","loved"]
negative_words=np.unique(np.asarray(negative_words)).tolist()
positive_words=np.unique(np.asarray(positive_words)).tolist()
min_len= len(negative_words) if len(negative_words)<len(positive_words) else len(positive_words)
negative_words=negative_words[0:min_len]
positive_words=positive_words[0:min_len]
# print("len(negative_words):"+str(len(negative_words)))
# print("len(positive_words):"+str(len(positive_words)))
# print("len(set(negative_words)):"+str(len(set(negative_words))))
# print("len(set(positive_words)):"+str(len(set(positive_words))))

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
academic_result['section'] = "science_and_art"

race_sterotypes_result = {}
race_sterotypes_result['target_x'] = black_male_names + black_female_names
race_sterotypes_result['target_y'] = caucasian_male_names + caucasian_female_names
race_sterotypes_result['attribute_a'] = black_steretypes
race_sterotypes_result['attribute_b'] = caucasian_stereotypes
race_sterotypes_result['section'] = "racial_stereotypes"


race_positive_negative = {}
race_positive_negative['target_x'] = black_male_names + black_female_names
race_positive_negative['target_y'] = caucasian_male_names + caucasian_female_names
race_positive_negative['attribute_a'] = negative_words
race_positive_negative['attribute_b'] = positive_words
race_positive_negative['section'] = "race_positive_negative"


results = [occupation_result, academic_result, race_sterotypes_result, race_positive_negative]



with open('sections.json', 'w') as f:
    json.dump(results, f)
