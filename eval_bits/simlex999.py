import argparse
import csv
import logging
import os

from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("embedding_filenames", help="The name and location of the word embedding")
parser.add_argument("pairs_filename", help="The name and location of pairs and locations")
parser.add_argument("result_filename", help="The name and location of the target result file")

args = parser.parse_args()
logging.info(args)

logging.info("Loading embeddings...")

embedding_filenames = args.embedding_filenames.split(',')
embeddings = []
for filename in embedding_filenames:
    embeddings.append(KeyedVectors.load(filename))

logging.info("Calculating results")
lines = [['Embedding', 'Pearson-corr-coeff', '2t-P-Value', 'Spearman-rank-order-corr-coeff', '2t-P-Value', 'oov-ratio']]
for i in range(0, len(embeddings)):
    embedding = embeddings[i]
    embedding_name = os.path.basename(embedding_filenames[i]).split('.')[0]
    result = embedding.wv.evaluate_word_pairs(args.pairs_filename)
    lines.append([embedding_name, result[0][0], result[0][1], result[1][0], result[1][1], result[2]])

logging.info('Writing results to target file')

output = open(args.result_filename, "w")
writer = csv.writer(output)
writer.writerows(lines)

output.close()
logging.info("Done.")
logging.info("Process complete.")
