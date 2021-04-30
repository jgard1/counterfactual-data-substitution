import logging
import random
from datetime import time
# import spacy
import sys

sys.path.append('./')
from utils import TwoWayDict  # src.utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Substitutor:

    def __init__(self, base_pairs, invert_cond, name_pairs=None, his_him=True):

        # logging.info("Loading spaCy model...")
        # self.nlp = spacy.load(spacy_model)
        # logging.info("Done.")

        # This flag tells it whether or not to apply the special case intervention to him/his/her/hers
        self.his_him = his_him
        self.invert_cond = invert_cond

        self.base_pairs = TwoWayDict()
        for (male, female) in base_pairs:
            self.base_pairs[male.lower()] = female.lower()

        self.name_pairs = TwoWayDict()
        for (male, female) in name_pairs:
            self.name_pairs[male.lower()] = female.lower()

    def probablistic_substitute(self, input_texts):
        for text in input_texts:
            if bool(random.getrandbits(1)):
                yield self.invert_document(text)
            else:
                yield text

    def invert_document(self, input_text):
        # Parse the doc
        # doc = self.nlp(input_text)

        flipped = None

        # invert sentences 100% of the time if control group (its 50% at the document level)
        # otherwise, invert sentences 50% of the time
        if bool(random.getrandbits(1)) or self.invert_cond == "invert_control":

            for idx, word_pos in enumerate(input_text):
                if self.invert_cond == "invert_word_names":
                    flipped = self.invert_word_names(word_pos)

                elif self.invert_cond == "invert_word_neutral":
                    flipped = self.invert_word_neutral(word_pos)

                else:  # self.invert_cond == "invert_control"
                    flipped = self.invert_word(word_pos)

                if flipped is not None:
                    input_text[idx][0] = flipped

        # # Walk through in reverse order making substitutions
        # for word in reversed(doc):
        #
        #     # Calculate inversion
        #     flipped = self.invert_word_neutral(word)  # invert_word(word)
        #
        #     if flipped is not None:
        #         # Splice it into output
        #         start_index = word.idx
        #         end_index = start_index + len(word.text)
        #         output = output[:start_index] + flipped + output[end_index:]

        return input_text

    def invert_word(self, word_pos):

        flipped = None

        word, pos = word_pos[0], word_pos[1]
        text = word.lower()

        # Handle base case
        if text in self.base_pairs.keys():
            flipped = self.base_pairs[text]

        # Handle name case
        elif text in self.name_pairs.keys():
            flipped = self.name_pairs[text]

        # Handle special case (his/his/her/hers)
        elif self.his_him:
            if text == "him":
                flipped = "her"
            elif text == "his":
                if pos == "NNS":
                    flipped = "hers"
                else:  # PRP/PRP$
                    flipped = "her"
            elif text == "her":
                if pos == "PRP$":
                    flipped = "his"
                else:  # PRP
                    flipped = "him"
            elif text == "hers":
                flipped = "his"

        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, word)
        return None

    def invert_word_neutral(self, word_pos):
        # invert_word_neutral_time = time.now()
        flipped = None
        word, pos = word_pos[0], word_pos[1]
        text = word.lower()

        # handle he/she case
        if text == "he" or text == "she":
            flipped = "they"

        # Handle base case
        elif text in self.base_pairs.keys():
            flipped = self.base_pairs[text]

        # Handle name case
        elif text in self.name_pairs.keys():
            flipped = self.name_pairs[text]

        # Handle special case (his/his/her/hers)
        elif self.his_him:
            if text == "him":
                flipped = "them"
            elif text == "his":
                if pos == "NNS":
                    flipped = "theirs"
                else:  # PRP$ (can't be PRP ??)
                    flipped = "their"

            elif text == "her":
                if pos == "PRP$":
                    flipped = "their"
                else:  # PRP
                    flipped = "them"
            elif text == "hers":
                flipped = "theirs"
        # print("invert_word time " + str(invert_word_neutral_time - time.now()))
        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, word)
        return None

    def invert_word_names(self, word_pos):

        flipped = None
        word, pos = word_pos[0], word_pos[1]
        text = word.lower()

        # handle he/she case
        if text == "he" or text == "she":
            flipped = "they"

        # Handle base case
        elif text in self.base_pairs.keys():
            flipped = self.base_pairs[text]

        # Handle name case
        elif text in self.name_pairs.keys():
            flipped = "NAME-PLACEHOLDER"

        # Handle special case (his/his/her/hers)
        elif self.his_him:
            if text == "him":
                flipped = "them"
            elif text == "his":
                if pos == "NNS":
                    flipped = "theirs"
                else:  # PRP$ (can't be PRP ??)
                    flipped = "their"

            elif text == "her":
                if pos == "PRP$":
                    flipped = "their"
                else:  # PRP
                    flipped = "them"
            elif text == "hers":
                flipped = "theirs"

        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, word)
        return None

    @staticmethod
    def match_case(input_string, target_string):
        # Matches the case of a target string to an input string
        # This is a very naive approach, but for most purposes it should be okay.
        if target_string.islower():
            return input_string.lower()
        elif target_string.isupper():
            return input_string.upper()
        elif target_string[0].isupper() and target_string[1:].islower():
            return input_string[0].upper() + input_string[1:].lower()
        else:
            # logging.warning("Unable to match case of {}".format(target_string))
            return input_string
