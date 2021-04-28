import os

from gensim.models import KeyedVectors
import numpy as np
from itertools import combinations
import argparse
import logging
import json
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class WordEmbeddingAssociationTest:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.cache_sim3 = dict()
        self.cache_similarity = dict()

    def individual(self, x, y, a, b):
        return ((self.memoised_similarity(x, a) -
                 self.memoised_similarity(x, b)) -
                (self.memoised_similarity(y, a) -
                 self.memoised_similarity(y, b)))

    def p_value(self, tar_x, tar_y, attr_a, attr_b):

        x_and_y = tar_x.union(tar_y)

        # Get all possible halves
        all_halves = combinations(x_and_y, (len(x_and_y) // 2))

        successes = 0

        sim4default = self.sim4(tar_x, tar_y, attr_a, attr_b)

        x = 0
        for half in all_halves:

            x += 1
            if x % 100 == 0:
                logging.info("Deterministic p-value on iteration {}".format(x))

            if (self.sim4(half, x_and_y.difference(half), attr_a, attr_b)) >= sim4default:
                successes += 1

        p_val = successes / x

        return p_val, successes, x

    def random_p_value(self, tar_x, tar_y, attr_a, attr_b, iterations):

        x_and_y = list(tar_x.union(tar_y))

        half_size = len(x_and_y) // 2

        successes = 0
        sim4default = self.sim4(tar_x, tar_y, attr_a, attr_b)

        for x in range(iterations):

            if x % 100 == 0:
                logging.info("Random p-value on iteration {}".format(x))

            perm = np.random.permutation(x_and_y)

            if (self.sim4(perm[:half_size], perm[half_size:], attr_a, attr_b)) >= sim4default:
                successes += 1

        p_val = (successes+1) / (iterations+1)

        return p_val, successes, iterations

    # Calculate a normalised measure of how separated the two distributions
    # (of associations between target and attribute) are
    def effect_size(self, tar_x, tar_y, attr_a, attr_b):

        total_x = 0
        for x in tar_x:
            total_x += self.memoised_sim3(x, attr_a, attr_b)
        mean_x = total_x / len(tar_x)

        total_y = 0
        for y in tar_y:
            total_y += self.memoised_sim3(y, attr_a, attr_b)
        mean_y = total_y / len(tar_y)

        tar_x_and_y = tar_x.union(tar_y)
        sim_x_and_y = set()
        for xy in tar_x_and_y:
            sim_x_and_y.add(self.memoised_sim3(xy, attr_a, attr_b))

        cohens_d = (mean_x - mean_y) / np.std(list(sim_x_and_y))
        return cohens_d

    # Measure the association of a word with the attribute
    def sim3(self, word, attr_a, attr_b):

        total_a = 0
        for a in attr_a:
            total_a += self.memoised_similarity(word, a)

        total_b = 0
        for b in attr_b:
            total_b += self.memoised_similarity(word, b)

        result = total_a / len(attr_a) - total_b / len(attr_b)
        return result

    # Measure the differential association of two sets of target words and an attribute
    def sim4(self, tar_x, tar_y, attr_a, attr_b):

        result = 0

        for x in tar_x:
            result += self.memoised_sim3(x, attr_a, attr_b)

        for y in tar_y:
            result -= self.memoised_sim3(y, attr_a, attr_b)

        return result

    # Memoised version of sim3
    def memoised_sim3(self, word, attr_a, attr_b):
        # Since sets are hashable we create a unique code:
        code = word + '  ' + ' '.join(attr_a) + '  ' + ' '.join(attr_a)
        if code in self.cache_sim3:
            return self.cache_sim3[code]
        # If not then calculate and store the result
        result = self.sim3(word, attr_a, attr_b)
        self.cache_sim3[code] = result
        return result

    # Memoised version of similarity
    def memoised_similarity(self, *args):
        if args in self.cache_similarity:
            return self.cache_similarity[args]
        # If not then calculate and store the result
        result = self.embeddings.similarity(*args)
        self.cache_similarity[args] = result
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filenames", help="The name and location of the word embedding")
    parser.add_argument("sections_filename", help="The name and location of the data json file")
    parser.add_argument("result_filename", help="The name and location of the target result file")

    args = parser.parse_args()
    logging.info(args)

    logging.info("Loading embeddings")

    embedding_filenames = args.embedding_filenames.split(',')
    embeddings = []
    for filename in embedding_filenames:
        embeddings.append(KeyedVectors.load(filename))

    with open(args.sections_filename, 'r') as file:
        data = file.read()

    result = json.loads(data)
    output = []

    # Lowercasify the targets and attributes
    temp = [x.lower() for x in result.get('target_x')]
    result['target_x'] = set(temp)
    temp = [x.lower() for x in result.get('target_y')]
    result['target_y'] = set(temp)
    temp = [x.lower() for x in result.get('attribute_a')]
    result['attribute_a'] = set(temp)
    temp = [x.lower() for x in result.get('attribute_b')]
    result['attribute_b'] = set(temp)

    cohens = []
    embedding_names = []

    for i in range(0, len(embeddings)):

        embedding = embeddings[i]
        embedding_name = os.path.basename(embedding_filenames[i]).split('.')[0]
        embedding_names.append(embedding_name)

        # Make a new test
        weat = WordEmbeddingAssociationTest(embedding)

        logging.info("Processing results for " + result.get('section'))

        assert (len(result.get('target_x')) == len(result.get('target_y')))
        assert (len(result.get('attribute_a')) == len(result.get('attribute_b')))

        X = result.get('target_x')
        Y = result.get('target_y')
        A = result.get('attribute_a')
        B = result.get('attribute_b')

        logging.info('Calculating effect-size')
        cohens.append(weat.effect_size(X,Y,A,B))

        out = [embedding_name]
        logging.info('Calculating results for monte carlo')
        for a in A:
            for b in B:
                for x in X:
                    for y in Y:
                        out.append(weat.individual(x,y,a,b))

        output.append(out)

    output = np.array(output).transpose()

    logging.info('Writing results to target file')

    weat_output = open(args.result_filename, "w")
    writer = csv.writer(weat_output)
    writer.writerows(output)

    logging.info("Embeddings: " + str(embedding_names))
    logging.info("Cohens:" + str(cohens))
    weat_output.close()
    logging.info("Done.")
    logging.info("Process complete.")
    return


if __name__ == '__main__':
    main()
