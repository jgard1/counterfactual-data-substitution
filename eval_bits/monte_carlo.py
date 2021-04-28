import argparse
import itertools
import logging
import csv
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("results", help="The input .csv results set")
parser.add_argument("--permutations", help="The number of permutations", type=int, default=10000)
parser.add_argument("output", help="The name and location of the target .csv result file")
args = parser.parse_args()
print(args)

results = []
R = args.permutations

logging.info("Loading results")
with open(args.results) as f:
    reader = csv.DictReader(f)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(float(value))
            except KeyError:
                data[header] = [float(value)]
embeddings = list(data.keys())

p_values = {}

for combo in itertools.combinations(embeddings, 2):

    logging.info("Performing perms test on {" + combo[0] + "} and {" + combo[1] + "}")

    s = 0
    pairwise = list(zip(data.get(combo[0]), data.get(combo[1])))

    # Calculate sample mean
    sample_a = 0
    sample_b = 0
    for pair in pairwise:
        sample_a += pair[0]
        sample_b += pair[1]

    effect = (sample_a-sample_b)/len(pairwise)
    sample_diff = abs(sample_a-sample_b) # We don't need to divide this to get the mean since the length is constant

    # Perform n permutations
    for i in range(R):

        if (i+1) % 1000 == 0:
            logging.info("On permutation " + str(i+1) + " out of " + str(R))

        test_a = 0
        test_b = 0

        # Add totals for random permutation
        for pair in pairwise:

            # Flip coin
            if random.getrandbits(1):
                test_a += pair[0]
                test_b += pair[1]
            else:
                test_a += pair[1]
                test_b += pair[0]

        test_diff = abs(test_a-test_b)

        if test_diff >= sample_diff:
            s += 1

    p = (s + 1)/(R + 1)

    p_values[combo] = (effect, p)

logging.info("Building results table")

results = [["EMBEDDINGS"] + embeddings]
# Build table
for row in embeddings:
    r = [row]
    # Generate empties
    for col in embeddings:
        if (col, row) in p_values.keys():
            result = p_values.get((col, row))
            eff = result[0]
            p = result[1]
            if p >= 0.05:
                r.append('=')
            elif p < 0.01:
                if eff > 0:
                    r.append('<<')
                else:
                    r.append('>>')
            else: # <0.05
                if eff > 0:
                    r.append('<')
                else:
                    r.append('>')
        else:
            r.append("")
    results.append(r)

logging.info("Writing")
with open(args.output, 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(results)

logging.info("Done")
