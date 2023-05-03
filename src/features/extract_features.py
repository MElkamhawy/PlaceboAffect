import numpy as np
from empath import Empath
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

lexicon = Empath()
BOW_FEATURE = "bag_of_words"
W2V_FEATURE = "word2vec"
EMPATH_FEATURE = "empath"
NGRAM_FEATURE = "n_grams"


class Vector:
    def __init__(self, name, text, config):
        """
        Initializes the Vector object.
        Args:
            name: "train", "dev", or "test", corresponding to the dataset.
            vectorizer: A Word2Vec model.
            text: A numpy array of strings.
            vector: A numpy array of vectors (numpy arrays).
            features_config: A dictionary containing the configuration for the features.
        """
        self.name = name
        self.text = text
        self.vectorizer = None
        self.vector = None
        self.features_config = config

    def process_features(self, vectorizer=None):
        """
        Calculates and concatenates feature vectors given an array of input text.
        Args:
            vectorizer: A Word2Vec model.
        """
        text_vectors = None
        upgraded_text = text_tokenize = [self._tokenize(element) for element in self.text]

        if self.features_config[BOW_FEATURE]:
            text_vectors = self._apply_bow(self.text, vectorizer)
        if self.features_config[NGRAM_FEATURE]:
            upgraded_text = self._apply_ngrams(self.text, tokenized_corpus=text_tokenize)
        if self.features_config[W2V_FEATURE]:
            text_vectors = self._apply_w2v(upgraded_text, vectorizer=vectorizer, text=text_tokenize)
        if self.features_config[EMPATH_FEATURE]:
            text_vectors = self._apply_empath(text_vectors)
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

    def _apply_empath(self, text_vectors):
        """
        Calculates empath scores for the input text and concatenates the resulting vector to the input text vectors.
        Args:
            text_vectors: A numpy array of vectors.
        Returns:
            A numpy array of vectors.
        """
        empath_vectors = [self._calculate_empath(element) for element in self.text]
        return np.concatenate((text_vectors, empath_vectors), axis=1)

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
    
    def _apply_ngrams(self, text, tokenized_corpus):
        # Create bi-grams and tri-grams
        vectorizer = CountVectorizer(ngram_range=(2, 4), analyzer="word")
        ngrams = vectorizer.fit_transform(text)
        ngram_feature_names = vectorizer.get_feature_names()

        # Train Word2Vec on tokenized_corpus and n-grams
        return tokenized_corpus + [ngram.split() for ngram in ngram_feature_names]

    def _apply_w2v(self, tokenized_corpus, vectorizer=None, text=None):
        """
        Uses a Word2Vec model to obtain the embedding for each token in the input text,
        then averages the embeddings for the entire text.
        Args:
            text: A string of input text.
            vectorizer: A Word2Vec model.
        Returns:
            A numpy array of vectors.
        """

        w2v_model = Word2Vec(
                    tokenized_corpus, min_count=1, vector_size=100, window=5
                )
        if vectorizer is None:
            self.vectorizer = w2v_model
        else:
            self.vectorizer = vectorizer

        # Apply the Word2Vec model to each word, then average for the entire text
        vectors = []
        for tweet in text:
            temp_vectors = []
            for word in tweet:
                try:
                    temp_vectors.append(self.vectorizer.wv[word])
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
