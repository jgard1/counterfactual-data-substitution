#!/usr/bin/env bash
EXP_NAME=$1
mkdir -p "../experiments/$EXP_NAME"
# "invert_word_names" "invert_word_neutral" "invert_control"
EXP_EMBEDDING_FILES="../embeddings/$EXP_NAME.wordvectors"
CTRL_EMBEDDING_FILE="../embeddings/invert_control.wordvectors"
DEFINITIONAL_PAIRS="../data/cda_default_pairs.json"

BIAS_CSV="../experiments/$EXP_NAME/bias.csv"
BIAS_LOG="../experiments/$EXP_NAME/bias_log.txt"
CLUSTERING_LOG="../experiments/$EXP_NAME/clustering_log.txt"

MIKOLOV_ANALOGIES="./family_analogies.txt"
MIKOLOV_CSV="../experiments/$EXP_NAME/mikolov.csv"
MIKOLOV_LOG="../experiments/$EXP_NAME/mikolov_log.txt"

SIMLEX_CSV="../experiments/$EXP_NAME/simlex.csv"
SIMLEX_LOG="../experiments/$EXP_NAME/simlex_log.txt"
SIMLEX_PAIRS="./simlex_clean.tsv"

WEAT_SECTIONS="./sections.json"
WEAT_CSV="../experiments/$EXP_NAME/weat.csv"
WEAT_LOG="../experiments/$EXP_NAME/weat_log.txt"

# BIAS_STATS="../experiments/$EXP_NAME/bias_stats.txt"
# BIAS_STATS_LOG="../experiments/$EXP_NAME/bias_stats_log.txt"

# MIKOLOV_STATS="../experiments/$EXP_NAME/mikolov_stats.txt"
# MIKOLOV_STATS_LOG="../experiments/$EXP_NAME/mikolov_stats_log.txt"


python bias_classifier.py $CTRL_EMBEDDING_FILE $EXP_EMBEDDING_FILES $DEFINITIONAL_PAIRS $BIAS_CSV $BIAS_LOG
python clustering_mean.py $CTRL_EMBEDDING_FILE $EXP_EMBEDDING_FILES $DEFINITIONAL_PAIRS $CLUSTERING_LOG
python mikolov_analogies_test.py $EXP_EMBEDDING_FILES $MIKOLOV_ANALOGIES $MIKOLOV_CSV $MIKOLOV_LOG
python simlex999.py $EXP_EMBEDDING_FILES $SIMLEX_PAIRS $SIMLEX_CSV $SIMLEX_LOG
# get statistical results on the bias output 
# python monte_carlo.py $BIAS_CSV $BIAS_STATS $BIAS_STATS_LOG
# python monte_carlo.py $MIKOLOV_CSV $MIKOLOV_STATS $MIKOLOV_STATS_LOG
python weat.py $EXP_EMBEDDING_FILES $WEAT_SECTIONS $WEAT_CSV $WEAT_LOG