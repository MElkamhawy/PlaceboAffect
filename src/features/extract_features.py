import numpy as np
import gensim

from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from empath import Empath
from sklearn.feature_extraction.text import CountVectorizer


lexicon = Empath()


class Vector:
    def __init__(self, name, text):
        """
        Initializes the Vector object.
        Args:
            name: "train", "dev", or "test", corresponding to the dataset.
            vectorizer: A Word2Vec model or CountVectorizer object
            text: A numpy array of strings.
            vector: A numpy array of vectors (numpy arrays).
        """
        self.name = name
        self.text = text
        self.vectorizer = None
        self.vector = None

    def process_features(self, strategy, vectorizer=None, empath=False):
        """
        Calculates and concatenates feature vectors given an array of input text.
        Args:
            strategy: a string indicating whether to use a 'BOW' countvectorizer or "W2V' word2vec model
            vectorizer: A Word2Vec model.
            empath: A boolean variable indicating whether or not to calculate empath scores.
        """

        if strategy == 'w2v':
            text_tokenize = [self._tokenize(element) for element in self.text]
            text_vectors = self._apply_w2v(text_tokenize, vectorizer)
            # If empath, apply empath to text and concatenate the resulting empath vector to the corresponding embedding
            if empath:
                empath_vectors = [self._calculate_empath(element) for element in self.text]
                text_vectors = np.concatenate((text_vectors, empath_vectors), axis=1)
        elif strategy == 'bow':
            text_vectors = self._apply_bow(self.text, vectorizer)

        else:
            raise ValueError(
                "strategy argument must be 'bow' for a bag of words approach or 'w2v' for a word2vec approach"
            )

        self.vector = text_vectors

    def _tokenize(self, text):
        """
        Tokenizes the input text via TweetTokenizer.
        Args:
            text: A string of input text.
        Returns:
            A list of tokens.
        """
        tweet_tokenizer = TweetTokenizer()
        tokens = tweet_tokenizer.tokenize(text)
        return tokens

    def _calculate_empath(self, text):
        """
        Calculates scores for 194 lexical categories for a given text string.
        Args:
            text: A string of input text.
        Returns:
            A list of values corresponding to each lexical category.
        """
        empath_dict = lexicon.analyze(text, normalize=True)
        empath_values = np.array(list(empath_dict.values()))
        return empath_values

    def _apply_w2v(self, text, vectorizer=None):
        """
        Uses a Word2Vec model to obtain the embedding for each token in the input text,
        then averages the embeddings for the entire text.
        Args:
            text: A string of input text.
            vectorizer: A Word2Vec model.
        Returns:
            A numpy array of vectors.
        """

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

        # Apply the Word2Vec model to each word, then average for the entire text
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

        return np.array(vectors)

    def _apply_bow(self, text, vectorizer=None):
        """
        Applies CountVectorizer to the input text, generating bag of words representations.
        Args:
            text: A numpy array of input strings.
            vectorizer: A CountVectorizer object. If None, creates a new object for training data and raises an error for dev or test data.
        Returns:
            A sparse matrix of bag of words representations.
        Raises:
            ValueError: If vectorizer is None and self.name is not 'train', or if self.name is not 'dev' or 'test'.
        """
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
