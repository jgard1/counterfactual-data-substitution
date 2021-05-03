import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import KeyedVectors

# Set up graphs
from sklearn.metrics import v_measure_score

TESTING = 1000

def normalise(vector):
    norm = np.linalg.norm(vector)
    # if norm == 0:
    # return vector
    return np.divide(vector, norm)


parser = argparse.ArgumentParser()
parser.add_argument("control_embedding_filename", help="The name and location of the control word embedding")
parser.add_argument("embedding_filenames", help="The name and location of the word embeddings")
parser.add_argument("definitional_pairs_filename", help="The name and location of pairs and locations")
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

control_m = control_embedding.most_similar(positive=[bias_direction], topn=TESTING//2)
control_f = control_embedding.most_similar(positive=[-bias_direction], topn=TESTING//2)

results = np.array([]).reshape(0, TESTING + 1)
v_measures = []
embedding_names = []

for j in range(0, len(embeddings)):

    embedding = embeddings[j]
    embedding_name = os.path.basename(embedding_filenames[j]).split('.')[0]
    embedding_names.append(embedding_name)
    f = []
    m = []

    ground_truth = []

    logging.info("Working out points")
    for word in control_m:
        ground_truth.append(0)
        if word[0] in embedding:
            m.append(embedding[word[0]])
        else:
            # Predict similar word
            m.append(bias_direction)

    for word in control_f:
        ground_truth.append(1)
        if word[0] in embedding:
            f.append(embedding[word[0]])
        else:
            # Predict similar word
            f.append(-bias_direction)

    a = f + m

    logging.info("Calculating tSNE")

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(a)
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = Y[:, 0]
    df_subset['tsne-2d-two'] = Y[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue="y",
        # palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    logging.info("saving plot")
    plt.savefig('/home/dalelee/counterfactual-data-substitution/experiments/invert_control/invert_control_tsne.png')

    logging.info("Computing samples")

    pairs = list(zip(Y, ground_truth))
    emb_v_measures = []
    for samples in range(1000):
        # Sample 100 elements
        sample = random.choices(pairs, k=200)
        points, truth = zip(*sample)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
        predicted = kmeans.labels_
        v_measure = v_measure_score(truth, predicted)
        emb_v_measures.append(v_measure)

    v_measures.append(emb_v_measures)

logging.info("Embeddings:" + str(embedding_names))
means = []
stds = []

for v in v_measures:
    means.append(np.mean(v))
    stds.append(np.std(v))
logging.info("V-Measures:" + str(means))
logging.info("V-Measures std:" + str(stds))
