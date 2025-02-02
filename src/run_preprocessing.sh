#!/usr/bin/env bash

# should point to counterfactual-data-substitution
BASE="/home/dalelee//counterfactual-data-substitution/"

# should point to tagged_wikicorpus 
DATA_SUBDIR="tagged_wikicorpus/"

# should never need to edit anything after here
ORIGINAL_DATA="$BASE$DATA_SUBDIR"
EMBEDDINGS="embeddings/"
mkdir -p $BASE$EMBEDDINGS

for COND in "invert_race"
do
	CORPUS_OUTDIR="$BASE$COND/"
	mkdir -p $CORPUS_OUTDIR
	python process_corpus.py $COND $ORIGINAL_DATA $CORPUS_OUTDIR
	
	EMBEDDINGS_OUTPATH="$BASE$EMBEDDINGS$COND"
	python gen_embeddings.py $CORPUS_OUTDIR $EMBEDDINGS_OUTPATH
done 
