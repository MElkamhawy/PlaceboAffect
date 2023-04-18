import numpy as np
from nltk.tokenize import TweetTokenizer
import gensim
from gensim.models import Word2Vec
from empath import Empath

lexicon = Empath()


class Vector:
    def __init__(self, name, text):
        """
        Initializes the Vector object.
        Args:
            name: "train", "dev", or "test", corresponding to the dataset.
            vectorizer: A Word2Vec model.
            text: A numpy array of strings.
            vector: A numpy array of vectors (numpy arrays).
        """
        self.name = name
        self.text = text
        self.vectorizer = None
        self.vector = None

    def process_features(self, vectorizer=None, empath=False):
        """
        Calculates and concatenates feature vectors given an array of input text.
        Args:
            vectorizer: A Word2Vec model.
            empath: A boolean variable indicating whether or not to calculate empath scores.
        """
        text_tokenize = [self._tokenize(element) for element in self.text]
        text_vectors = self._vectorize_text(text_tokenize, vectorizer)
        # If empath, apply empath to text and concatenate the resulting empath vector to the corresponding embedding
        if empath:
            empath_vectors = [self._calculate_empath(element) for element in self.text]
            text_vectors = np.concatenate((text_vectors, empath_vectors), axis=1)
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

    def _vectorize_text(self, text, vectorizer=None):
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
