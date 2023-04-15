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


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name  # train, dev, or test
        self.vectorizer = None
        self.text = None
        self.label = None
        self.vector = None

    @classmethod
    def from_csv(cls, train_path, name):
        return cls(pd.read_csv(train_path), name)

    def process(self, text_name, target_name, vectorizer=None, empath=False):
        # Convert target into array of labels
        self.label = self._process_target(target_name)
        # Preprocess text data
        self.text = self._process_text(text_name)
        self.vector = self._process_features(vectorizer, empath)

    def _process_target(self, target_name):
        return np.array(self.raw_df[target_name])

    def _process_text(self, text_name):
        text_clean = np.array(
            [self._clean_text(element) for element in self.raw_df[text_name]]
        )
        return text_clean
    
    def _process_features(self, vectorizer=None, empath=False):
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

    def _lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def _remove_stopwords(self, tokens):
        stop_words = set(stopwords.words("english"))
        return [token for token in tokens if not token in stop_words]

    def _extract_hashtags(self, text):
        hashtags = re.findall(r"#[A-Za-z0-9_]+", text)
        hashtag_tokens_combined = []
        for hashtag in hashtags:
            hashtag_tokens = re.findall(r"[A-Z]+[a-z]*", hashtag[1:])
            hashtag_tokens_combined.extend(hashtag_tokens)
        return [token.lower() for token in hashtag_tokens_combined]

    def _clean_text(self, original_text):
        """
        Clean up the description: lowercase, remove brackets, remove various characters
        TODO: Determine how much of this is actually necessary.
            This is pretty standard preprocessing, but not necessarily how we want to treat tweets.
        """
        hashtag_tokens = self._extract_hashtags(original_text)
        text = str(original_text).lower()  # Convert to lowercase
        text = re.sub(r"https?://\S+", " ", text)  # Remove URLs
        text = re.sub(r"[#@][A-Za-z0-9_]+", " ", text)  # Remove hashtags and mentions
        text = re.sub(
            r"[\W\d]+", " ", text
        ).strip()  # Remove non-word characters, numbers, and extra whitespaces
        tokens = self._tokenize(text)
        lemmatized_tokens = self._lemmatize(tokens)
        filtered_tokens = self._remove_stopwords(lemmatized_tokens)
        filtered_tokens.extend(hashtag_tokens)
        clean_text = " ".join(filtered_tokens)
        if clean_text == "":
            clean_text = str(original_text).lower()
        return clean_text
    
    def _calculate_empath(self, text):
        lexicon = Empath()
        empath_dict = lexicon.analyze(text, normalize=True)
        empath_values = np.array(list(empath_dict.values()))
        return empath_values
    
    def _vectorize_text(self, text, vectorizer=None):

        # Use an existing vectorizer for the training data.
        if vectorizer is None:
            if self.name == "train":
                vectorizer = gensim.models.Word2Vec(text, min_count = 1, vector_size = 100, window = 5) # CBOW model
                self.vectorizer = vectorizer
            else:
                raise ValueError("vectorizer cannot be None when self.name is not 'train'")

        # Apply the Word2Vec model to each word, then average for the entire tweet
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
