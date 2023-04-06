import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

class Data:
    def __init__(self, train_raw, dev_raw, test_raw):
        self.train_raw = train_raw
        self.dev_raw = dev_raw
        self.test_raw = test_raw
        self.y_train = None
        self.X_train = None
        self.y_dev = None
        self.X_dev = None
        self.y_test = None
        self.X_test = None

    @classmethod
    def from_csv(cls, train_path, dev_path, test_path):
        return cls(pd.read_csv(train_path), pd.read_csv(dev_path), pd.read_csv(test_path))

    def process(self):
        # Set X Arrays
        self.y_train = self.train_raw['HS']
        self.y_dev = self.dev_raw['HS']
        self.y_test = self.test_raw['HS']

        # Create TF-IDF vectorizer
        self.X_train = self._vectorize(set='train', df=self.train_raw, text_column='text')
        self.X_dev = self._vectorize(set='dev', df=self.dev_raw, text_column='text')
        self.X_test = self._vectorize(set='test', df=self.test_raw, text_column='text')
        print("proces completed")

    def _vectorize(self, set, df, text_column):
        """

        :param set: a string indicating which split of the data the function is being applied to
        :param df: the hateval dataframe read in from a csv
        :param text_column: the name of the column with text to be processed
        :return: tfidf sparse matrix or numpy array
        """

        vectorizer = TfidfVectorizer()
        print(set)
        print(df[text_column])
        if set == 'train':
            tfidf = vectorizer.fit_transform(df[text_column])
        else:
            tfidf = vectorizer.transform(df[text_column])

        return tfidf



