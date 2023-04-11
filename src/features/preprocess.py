import re
import string
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name
        self.vectorizer = None
        self.text = None
        self.label = None

    @classmethod
    def from_csv(cls, train_path, name):
        return cls(pd.read_csv(train_path), name)

    def process(self, text_name, target_name, vectorizor=None):
        # TODO Document
        # Convert target into array of labels
        self.label = self._process_target(target_name)

        # Preprocess text data
        self.text = self._process_text(text_name, vectorizor)

    def _process_target(self, target_name):
        return self.raw_df[target_name]

    def _process_text(self, text_name, vectorizor=None):
        # TODO Document
        text_clean = np.array([self._clean_text(element) for element in self.raw_df[text_name]])
        text_vector = self._vectorize_text(text_clean, vectorizor)
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

    def _vectorize_text(self, text, vectorizer=None):
        # TODO Build this out. Should be num instances x vocabulary shape numpy array.d
        #   Probably doesn't make sense to use CountVectorizer, given that we want some flexibility with embeddings.

        # On the text set, use an existing vectorizer for the training data.
        if vectorizer is None:
            if self.name == 'train':
                vectorizer = CountVectorizer()
            else:
                raise ValueError("vectorizer cannot be None when self.name is not 'train'")

        # Fit and transform the data for training data
        if self.name == 'train':
            text_matrix = vectorizer.fit_transform(text)

            # Save Vectorizer Object for use with dev/test data
            self.vectorizer = vectorizer

        # Only transform the data for dev or test data
        elif self.name == 'dev' or self.name == 'test':
            text_matrix = vectorizer.transform(text)

        return text_matrix

    def _apply_embeddings(self, text):
        # TODO Apply pretrained embeddings
        # Word2Vec? Glove?
        return text
