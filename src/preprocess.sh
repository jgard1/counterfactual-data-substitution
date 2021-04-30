#!/usr/bin/env bash
python3 process_corpus.py invert_word_names ~/counterfactual-data-substitution/tagged_wikicorpus/ ~/counterfactual-data-substitution/invert_names/
python3 process_corpus.py invert_word_neutral ~/counterfactual-data-substitution/tagged_wikicorpus/ ~/counterfactual-data-substitution/invert_neutral/
python3 process_corpus.py invert_control ~/counterfactual-data-substitution/tagged_wikicorpus/ ~/counterfactual-data-substitution/invert_control/