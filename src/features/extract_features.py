import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec

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

    def _tokenize(self, text):
        tweet_tokenizer = TweetTokenizer()
        tokens = tweet_tokenizer.tokenize(text)
        return tokens
        
    def _vectorize_text(self, text, vectorizer=None):
        # TODO Build this out. Should be num instances x vocabulary shape numpy array.d
        #   Probably doesn't make sense to use CountVectorizer, given that we want some flexibility with embeddings.

        # On the text set, use an existing vectorizer for the training data.
        if vectorizer is None:
            if self.name == "train":
                vectorizer = gensim.models.Word2Vec(text, min_count = 1, vector_size = 100, window = 5) # CBOW model
                self.vectorizer = vectorizer
            else:
                raise ValueError("vectorizer cannot be None when self.name is not 'train'")

        # apply the Word2Vec model to each word, then average for the entire tweet
        vectors = []
        for tweet in text:
            temp_vectors = []
            for word in tweet:
                try:
                    temp_vectors.append(vectorizer.wv[word])
                except:
                    temp_vectors.append([0]*100)
            average_vector = np.mean(temp_vectors, axis=0)
            vectors.append(average_vector)

        return vectors

    def process_features(self, text_name, vectorizer=None):
        # TODO Document
        text_tokenize = [self._tokenize(element) for element in df[text_name]]
        text_vectors = self._vectorize_text(text_tokenize, vectorizor)
        return text_vectors
