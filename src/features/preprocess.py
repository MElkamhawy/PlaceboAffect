import re
import string
import pandas as pd
import numpy as np


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name
        self.text = None
        self.label = None

    @classmethod
    def from_csv(cls, train_path, name):
        return cls(pd.read_csv(train_path), name)

    def process(self, text_name, target_name):
        # TODO Document
        # Convert target into array of labels
        self.label = self._process_target(target_name)

        # Preprocess text data
        self.text = self._process_text(text_name)

    def _process_target(self, target_name):
        return self.raw_df[target_name]

    def _process_text(self, text_name):
        # TODO Document
        text_clean = np.array([self._clean_text(element) for element in self.raw_df[text_name]])
        text_vector = self._vectorize_text(text_clean)
        text_embeddings = self._apply_embeddings(text_vector)
        return text_embeddings

    def _clean_text(self, text):
        """
        Clean up the description: lowercase, remove brackets, remove various characters
        TODO: Determine how much of this is actually necessary.
            This is pretty standard preprocessing, but not necessarily how we want to treat tweets.
        """
        text = str(text)
        text = text.lower()
        text = re.sub(r'/', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r"\s+", ' ', text).strip()
        return text

    def _vectorize_text(self, text):
        # TODO Build this out. Should be num instances x vocabulary shape numpy array.
        return text

    def _apply_embeddings(self, text):
        # TODO Apply pretrained embeddings
        # Word2Vec? Glove?
        return text
