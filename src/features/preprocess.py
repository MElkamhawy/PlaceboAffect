import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import emoji
import contractions


class Data:
    def __init__(self, raw_df, name):
        self.raw_df = raw_df
        self.name = name
        self.text = None
        self.label = None
        self.remove_stopwords = False
        self.remove_numbers = False
        self.remove_mentions = False
        self.negation_words = [
            "no",
            "not",
            "never",
            "none",
            "neither",
            "nor",
            "nothing",
            "nobody",
            "nowhere",
            "cannot",
            "aint",
            "wont",
            "didnt",
            "barely",
            "without",
        ]

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
            [self._process_line(element) for element in self.raw_df[text_name]]
        )
        return text_clean

    def _handle_negation(self, text):
        """
        Adds a "NEG_" prefix to the word following negation words.

        Args:
            text (str): The text string to handle negation in.

        Return:
            str: The text with negation handled.
        """
        processed_words = []
        negate_next_word = False
        for word in text.split():
            if word in self.negation_words:
                negate_next_word = True
            elif negate_next_word:
                word = f"NEG_{word}"
                negate_next_word = False
            processed_words.append(word)
        processed_text = " ".join(processed_words)
        return processed_text

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
        return [
            token
            for token in tokens
            if not token in stop_words or token in self.negation_words
        ]

    def _extract_hashtags_and_mentions(self, text):
        """
        Extract hashtags mentions from the input text and replace them with processed versions.

        Args:
            text (str): Input string to extract mentions from.

        Returns:
            str: The modified string with mentions replaced by their processed versions.
        """
        if self.remove_mentions:
            tags = re.findall(r"[#][A-Za-z0-9_]+", text)
            text = re.sub(r"[@][A-Za-z0-9_]+", " ", text)
        else:
            tags = re.findall(r"[@#][A-Za-z0-9_]+", text)
        processed_text = text
        for tag in tags:
            processed_tag = self._process_tag(tag)
            processed_text = processed_text.replace(tag, processed_tag)
        return processed_text

    def _process_tag(self, tag):
        """
        Process a given mention string to remove "@" symbol and split based on underscore or capitalization.

        Args:
            mention (str): Mention string to process.

        Returns:
            str: The processed version of the mention.
        """
        tag = tag[1].upper() + tag[2:]
        # Split the mention into tokens based on underscore or capitalization
        if "_" in tag:
            tokens_split_by_underscore = tag.split("_")
            processed_tag = " ".join(tokens_split_by_underscore)
        else:
            tokens_split_by_capitalization = re.findall(r"[A-Z]+[a-z]*", tag)
            processed_tag = " ".join(tokens_split_by_capitalization)
        return processed_tag.lower()

    def _replace_emoji_with_words(self, text):
        """
        Replaces emojis with their corresponding words using the emoji module.

        Args:
            text (str): The text string to replace emojis in.

        Return:
            str: The text with emojis replaced with words.
        """
        text = emoji.demojize(text)
        text = re.sub(r"_", " ", text)
        return text

    def _expand_contractions(self, text):
        """
        Expands contractions in the given text.

        Args:
            text (str): The text string to expand contractions in.

        Return:
            str: The text with contractions expanded.
        """
        return contractions.fix(text)

    def _replace_curse_words(self, text):
        """
        Replaces curse words in the given text with their corresponding original form.

        Args:
            text (str): The text string to replace curse words in.

        Return:
            str: The text with curse words replaced with their corresponding original form.
        """
        curse_words = {
            r"\bstfu\b": "shut the fuck up",
            r"\bwtf\b": "what the fuck",
            r"\bf[uck*]+ing?\b": "fucking",
            r"\bf[uck*]+er\b": "fucker",
            r"\bf[uck*]+\b": "fuck",
            r"\bb[i*][t*][c*]h\b": "bitch",
            r"sh[1!-*]t": "shit",
            r"a\*\*": "ass",
        }
        for key, value in curse_words.items():
            curse_word_regex = re.compile(key, re.IGNORECASE)
            text = curse_word_regex.sub(value, text)
        return text

    def _extract_useful_information(self, original_text):
        """
        Extracts useful information from the given text by processing hashtags, emojis, contractions, negation, and curse words.

        Args:
            original_text (str): The raw text string to be processed.

        Return:
            str: The processed text with useful information extracted.
        """
        text = self._extract_hashtags_and_mentions(original_text)
        text = self._replace_emoji_with_words(text)
        text = self._replace_curse_words(text)
        text = self._expand_contractions(text)
        text = text.lower()
        text = self._handle_negation(text)
        return text

    def _clean_text(self, original_text):
        """
        Cleans the text by removing URLs, HTML characters, unicode characters, special characters,
        and extra whitespaces. Can also remove numbers if specified.

        Args:
            original_text (str): The raw text string to be cleaned.

        Return:
            str: The cleaned text.
        """
        text = str(original_text.encode("utf-8"))  # Parse out the unicode characters
        text = re.sub(
            r"^b'?", "", text
        )  # Remove byte designator at beginning of the line
        text = re.sub("'$", "", text)  # Remove byte designator at end of the line
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"&[^;\s]+;", "", text)  # Remove html characters LIKE &amp;
        text = re.sub(r"\\x\S+", "", text)  # Remove unicode characters
        text = re.sub(r"'\bs\b", "", text)
        text = re.sub(r"[^\w\s]+", " ", text)  # Remove non-word characters and numbers
        if self.remove_numbers:
            text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text)
            text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)  # Remove extra whitespaces
        return text

    def _prepare_text(self, original_text):
        """
        Prepares the given text by tokenizing, lemmatizing, and removing stop words(optional).

        Args:
            original_text (str): The text string to be prepared.

        Return:
            str: The cleaned and prepared text.
        """
        tokens = self._tokenize(original_text)
        lemmatized_tokens = self._lemmatize(tokens)
        if self.remove_stopwords:
            lemmatized_tokens = self._remove_stopwords(lemmatized_tokens)
        text = " ".join(lemmatized_tokens)
        return text

    def _process_line(self, original_text):
        """
        Processes a single line of text by extracting useful information, cleaning and preparing the text.

        Args:
            original_text (str): The raw text string to be processed.

        Return:
            str: The cleaned and prepared text.
        """
        text = self._extract_useful_information(original_text)
        clean_text = self._clean_text(text)
        processed_text = self._prepare_text(clean_text)
        if processed_text == "":
            processed_text = str(original_text).lower()
        return processed_text
