import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name
        self.text = None
        self.label = None

    @classmethod
    def from_csv(cls, train_path, name):
        """
        Create an instance of the class from a CSV file.

        Args:
            train_path (str): The file path of the CSV file to read.
            name (str): The name of the instance (train, dev or test).

        Returns:
            cls: An instance of the class initialized with data from the CSV file.
        """
        return cls(pd.read_csv(train_path), name)

    def process(self, text_name, target_name):
        """
        Process the text data and target labels.

        Args:
            text_name (str): The name of the text column in the raw data.
            target_name (str): The name of the target column in the raw data.
        """
        self.label = self._process_target(target_name)
        self.text = self._process_text(text_name)

    def _process_target(self, target_name):
        """
        Process the target column in the raw data.

        Args:
            target_name (str): The name of the target column in the raw data.

        Returns:
            np.ndarray: A numpy array of target labels.
        """
        return np.array(self.raw_df[target_name])

    def _process_text(self, text_name):
        """
        Process the text column in the raw data.

        Args:
            text_name (str): The name of the text column in the raw data.

        Returns:
            np.ndarray: A numpy array of cleaned text data.
        """
        text_clean = np.array(
            [self._clean_text(element) for element in self.raw_df[text_name]]
        )
        return text_clean

    def _tokenize(self, text):
        """
        Tokenize the input string using nltk.TweetTokenizer()

        Args:
            text (str): Input string to tokenize

        Returns:
            list: A list of lemmatized tokens
        """
        tweet_tokenizer = TweetTokenizer()
        tokens = tweet_tokenizer.tokenize(text)
        return tokens

    def _lemmatize(self, tokens):
        """
        Lemmatize the input tokens.

        Args:
            tokens (list): A list of tokens to lemmatize.

        Returns:
            list: A list of lemmatized tokens.
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def _remove_stopwords(self, tokens):
        """
        Remove stopwords from the input tokens.

        Args:
            tokens (list): A list of tokens to remove stopwords from.

        Return:
            list: A list of tokens with stopwords removed.
        """
        stop_words = set(stopwords.words("english"))
        return [token for token in tokens if not token in stop_words]

    def _extract_hashtags(self, text):
        """
        Extract hashtags from the input text and split on capital letters.

        Args:
            text (str): Input string to to extract hashtags from.

        Returns:
            list: A list of extracted hashtag tokens in lowercase.
        """
        hashtags = re.findall(r"#[A-Za-z0-9_]+", text)
        hashtag_tokens_combined = []
        for hashtag in hashtags:
            hashtag_tokens = re.findall(r"[A-Z]+[a-z]*", hashtag[1:])
            hashtag_tokens_combined.extend(hashtag_tokens)
        return [token.lower() for token in hashtag_tokens_combined]

    def _clean_text(self, original_text):
        """
        Clean and prepare text: extract and process hashtags, convert to lowercase,
        remove URLs, mentions, remove special characters.

        Args:
            original_txt (str): The raw text string to be cleaned.

        Return:
            str: The cleaned text.
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
