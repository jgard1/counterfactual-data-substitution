####################################
### MODIFIED BY JOSH################
####ERROR MEMES POSSIBLE############
####################################


import argparse
import csv
import logging
import os
import random

import numpy as np
from sklearn.decomposition import PCA
import json
from sklearn.svm import SVC

from gensim.models import KeyedVectors

TESTING = 5000
SUBSET = 1000

def normalise(vector):
    norm = np.linalg.norm(vector)
    # if norm == 0:
    # return vector
    return np.divide(vector, norm)




parser = argparse.ArgumentParser()
parser.add_argument("control_embedding_filename", help="The name and location of the control word embedding")
parser.add_argument("embedding_filenames", help="The name and location of the word embeddings")
parser.add_argument("definitional_pairs_filename", help="The name and location of pairs and locations")
parser.add_argument("result_filename", help="The name and location of pairs and locations")
parser.add_argument("log_filename", help="The file logs are ritten to")

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=args.log_filename)

logging.info(args)

logging.info("Loading control embedding")
control_embedding = KeyedVectors.load(args.control_embedding_filename)

logging.info("Loading embeddings...")
embedding_filenames = args.embedding_filenames.split(',')
embeddings = []
for filename in embedding_filenames:
    embeddings.append(KeyedVectors.load(filename))
# embeddings.append(KeyedVectors.load(embedding_filenames[0]))

with open(args.definitional_pairs_filename, "r") as f:
    definitional_gender_pairs = json.load(f)
logging.info("{} gender-definitional pairs loaded.".format(len(definitional_gender_pairs)))

samples = []

logging.info("Calculating bias direction")
for [a, b] in definitional_gender_pairs:
    if a in control_embedding and b in control_embedding: 
        a = normalise(control_embedding[a])
        b = normalise(control_embedding[b])
        mean = np.divide(np.add(a, b), 2)
        samples.append(np.subtract(a, mean).tolist())
        samples.append(np.subtract(b, mean).tolist())
pca = PCA(n_components=1)
pca.fit(samples)
bias_direction = pca.components_[0]

# Set most biased words
control_m = control_embedding.most_similar(positive=[bias_direction], topn=TESTING // 2)
control_f = control_embedding.most_similar(positive=[-bias_direction], topn=TESTING // 2)

# Remove words not in all embeddings

# semantically this is ierating over embedding sets not individual embedddings 
for j in range(0, len(embeddings)):
    embedding = embeddings[j]
    removed_words_m = []
    removed_words_f = []
    for word in control_m:
        if word[0] not in embedding:
            removed_words_m.append(word)
            TESTING -= 1
            logging.info("Removed {} from male set".format(word[0]))
    for word in removed_words_m:
        control_m.remove(word)
    for word in control_f:
        if word[0] not in embedding:
            removed_words_f.append(word)
            TESTING -= 1
            logging.info("Removed {} from female set".format(word[0]))
    for word in removed_words_f:
        control_f.remove(word)

# Shuffle
random.shuffle(control_f)
random.shuffle(control_m)

results = np.array([]).reshape(0, TESTING + 1 - SUBSET)
percents = []
embedding_names = []

# For each embedding set we are comparing
for j in range(0, len(embeddings)):

    embedding = embeddings[j]
    embedding_name = os.path.basename(embedding_filenames[j]).split('.')[0]
    embedding_names.append(embedding_name)
    f = []
    m = []

    logging.info("Working out data")

    for word in control_m:
        if word[0] in embedding:
            m.append((embedding[word[0]], False))
        else:
            logging.warning("{} not present in {} and not successfully removed".format(word[0], embedding_name))

    for word in control_f:
        if word[0] in embedding:
            f.append((embedding[word[0]], True))
        else:
            logging.warning("{} not present in {} and not successfully removed".format(word[0], embedding_name))


    logging.info("Calculating training and test sets")

    training = f[:(SUBSET//2)] + m[:(SUBSET//2)]
    test = f[(SUBSET//2):] + m[(SUBSET//2):]
    training_data, training_correct = zip(*training)
    test_data, test_correct = zip(*test)

    training_data = list(training_data)
    training_correct = list(training_correct)
    test_data = list(test_data)
    test_correct = list(test_correct)

    logging.info("Fitting classifier")

    clf = SVC()
    clf.fit(training_data, training_correct)

    logging.info("Running classifier")
    test_result = clf.predict(test_data)

    logging.info("Calculating results")
    correct = np.logical_xor(test_correct, test_result).astype(int)

    percents.append(100 * (1 - (sum(correct) / len(correct))))

    results = np.append(results, [np.append([embedding_name], correct)], axis=0)


results = results.transpose()

logging.info("\n\n\n\nEmbeddings:" + str(embedding_names))
logging.info("Percents:" + str(percents))


logging.info("writing output")
output = open(args.result_filename, "w+")
writer = csv.writer(output)
writer.writerows(results)

output.close()
logging.info("Done.")
logging.info("Process complete.")
