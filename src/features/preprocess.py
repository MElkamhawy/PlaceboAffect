import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name  # train, dev, or test
        self.vectorizer = None
        self.text = None
        self.label = None

    @classmethod
    def from_csv(cls, train_path, name):
        return cls(pd.read_csv(train_path), name)

    def process(self, text_name, target_name, vectorizor=None):
        # Convert target into array of labels
        self.label = self._process_target(target_name)
        # Preprocess text data
        self.text = self._process_text(text_name, vectorizor)

    def _process_target(self, target_name):
        return np.array(self.raw_df[target_name])

    def _process_text(self, text_name, vectorizor=None):
        text_clean = np.array(
            [self._clean_text(element) for element in self.raw_df[text_name]]
        )
        text_vector = self._vectorize_text(text_clean, vectorizor)
        text_embeddings = self._apply_embeddings(text_vector)
        return text_embeddings

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

    def _clean_text(self, text):
        """
        Clean up the description: lowercase, remove brackets, remove various characters
        TODO: Determine how much of this is actually necessary.
            This is pretty standard preprocessing, but not necessarily how we want to treat tweets.
        """
        hashtag_tokens = self._extract_hashtags(text)
        text = str(text).lower()  # Convert to lowercase
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
        return clean_text

    def _vectorize_text(self, text, vectorizer=None):
        # TODO Build this out. Should be num instances x vocabulary shape numpy array.d
        #   Probably doesn't make sense to use CountVectorizer, given that we want some flexibility with embeddings.

        # On the text set, use an existing vectorizer for the training data.
        if vectorizer is None:
            if self.name == "train":
                vectorizer = CountVectorizer()
            else:
                raise ValueError(
                    "vectorizer cannot be None when self.name is not 'train'"
                )

        # Fit and transform the data for training data
        if self.name == "train":
            text_matrix = vectorizer.fit_transform(text)

            # Save Vectorizer Object for use with dev/test data
            self.vectorizer = vectorizer

        # Only transform the data for dev or test data
        elif self.name == "dev" or self.name == "test":
            text_matrix = vectorizer.transform(text)

        return text_matrix

    def _apply_embeddings(self, text):
        # TODO Apply pretrained embeddings
        # Word2Vec? Glove?
        return text
