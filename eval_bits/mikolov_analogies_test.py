import argparse
import logging
import os
import numpy as np

from gensim.models import KeyedVectors
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("embedding_filenames", help="The name and location of the word embedding")
parser.add_argument("analogies_filename", help="The name and location of the analogies .txt")
parser.add_argument("result_filename", help="The name and location of the target .csv result file")
args = parser.parse_args()
print(args)

logging.info("Loading embeddings")
embedding_filenames = args.embedding_filenames.split(',')
embeddings = []
for filename in embedding_filenames:
    embeddings.append(KeyedVectors.load(filename))

logging.info("Opening tests")
analogies = []
with open(args.analogies_filename) as f:
    r = csv.reader(f, delimiter=' ')
    for row in r:
        analogies.append(row)

logging.info("Calculating results")
results = []
means = []
embedding_names = []
for i in range(0, len(embeddings)):
    embedding = embeddings[i].wv
    embedding_name = os.path.basename(embedding_filenames[i]).split('.')[0]
    embedding_names.append(embedding_name)
    result_set = [embedding_name]
    for a in analogies:
        result_set.append(
            int(embedding.most_similar(positive=[a[1], a[2]], negative=[a[0]])[0][0] == a[3]))
    results.append(result_set)
    means.append(np.mean(result_set[1:]))

results = np.array(results)
results = results.transpose()

logging.info('Embeddings: ' + str(embedding_names))
logging.info('Means: ' + str(means))

logging.info('Writing results to target file')
output = open(args.result_filename, "w")
writer = csv.writer(output)
writer.writerows(results)

output.close()
logging.info("Done.")
logging.info("Process complete.")
