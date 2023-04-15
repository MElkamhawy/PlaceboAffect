import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models import Word2Vec
from empath import Empath


class FeatureVector:
    def __init__(self, text, name):
        self.text = text
        self.name = name
        self.vectorizer = None
        self.label = None
        self.vector = None

    def process_features(self, vectorizer=None, empath=False):
        text_tokenize = [self._tokenize(element) for element in self.text]
        text_vectors = self._vectorize_text(text_tokenize, vectorizer)
        # if empath, apply empath to text
        if empath:
            empath_vectors = [self._calculate_empath(element) for element in self.text]
            text_vectors = np.concatenate((text_vectors, empath_vectors), axis=1)
        return text_vectors

    def _tokenize(self, text):
        tweet_tokenizer = TweetTokenizer()
        tokens = tweet_tokenizer.tokenize(text)
        return tokens

    def _calculate_empath(self, text):
        lexicon = Empath()
        empath_dict = lexicon.analyze(text, normalize=True)
        empath_values = np.array(list(empath_dict.values()))
        return empath_values

    def _vectorize_text(self, text, vectorizer=None):
        # Use an existing vectorizer for the dev data.
        if vectorizer is None:
            if self.name == "train":
                vectorizer = gensim.models.Word2Vec(
                    text, min_count=1, vector_size=100, window=5
                )  # CBOW model
                self.vectorizer = vectorizer
            else:
                raise ValueError(
                    "vectorizer cannot be None when self.name is not 'train'"
                )

        # Apply the Word2Vec model to each word, then average for the entire tweet
        vectors = []
        for tweet in text:
            temp_vectors = []
            for word in tweet:
                try:
                    temp_vectors.append(vectorizer.wv[word])
                except:
                    temp_vectors.append([0] * 100)
            average_vector = np.mean(temp_vectors, axis=0)
            vectors.append(average_vector)

        return
