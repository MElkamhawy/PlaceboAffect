import spacy
import re
from unidecode import unidecode
from nltk.tokenize import TweetTokenizer

lines = [
    "Me estoy comiendo la picada árabe más rica de mi vida",
    "@Haryachyzaychyk Callate zorra y mama duro! 😍",
    "MMMMM TU NADA MÁS KERIAS ENSEÑAR LAS CHCHIS PINCHE BIEJA PUTA!!!!",
]

sentence = "gtfo #gtfoo gtfomy country"


def remove_accent(text):
    return unidecode(text)


def tokenize_nltk(text):
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text, language="spanish")
    return tokens


def main():
    nlp = spacy.load("es_core_news_sm")
    for text in lines:
        doc = nlp(text)
        lemmatized_tokens = [token.text for token in doc]
        print(lemmatized_tokens)
        print(tokenize_nltk(text))


main()
