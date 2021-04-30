import argparse
import logging
import os
import numpy as np

from gensim.models import KeyedVectors
import csv

parser = argparse.ArgumentParser()
parser.add_argument("embedding_filenames", help="The name and location of the word embedding")
parser.add_argument("analogies_filename", help="The name and location of the analogies .txt")
parser.add_argument("result_filename", help="The name and location of the target .csv result file")
parser.add_argument("log_filename", help="The file logs are ritten to")

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=args.log_filename)
logging.info(args)

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
    embedding = embeddings[i]
    embedding_name = os.path.basename(embedding_filenames[i]).split('.')[0]
    embedding_names.append(embedding_name)
    result_set = [embedding_name]
    for a in analogies:
        if(a[0] in embedding and a[1] in embedding and a[2] in embedding and a[3] in embedding):
            result_set.append(
                int(embedding.most_similar(positive=[a[1], a[2]], negative=[a[0]])[0][0] == a[3]))
            # print("appending stuff")
    results.append(result_set)
    means.append(np.mean(result_set[1:]))

# print(results)
results = np.array(results)
results = results.transpose()
# print(results)

logging.info('Embeddings: ' + str(embedding_names))
logging.info('Means: ' + str(means))

logging.info('Writing results to target file')
output = open(args.result_filename, "w")
writer = csv.writer(output)
writer.writerows(results)

output.close()
logging.info("Done.")
logging.info("Process complete.")
