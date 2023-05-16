import spacy
import re
from unidecode import unidecode
from nltk.tokenize import TweetTokenizer

lines = [
    "Me estoy comiendo la picada árabe más rica de mi vida",
    "@Haryachyzaychyk Callate zorra y mama duro! 😍",
    "Acabo de escuchar a Casado diciendo que hay DECENAS DE MILLONES de subsaharianos ahora mismo reuniendo dinero para venir a Europa. No sé qué me asusta más, que este idiota diga esas cosas o que haya tantos tarados deseando creérselas.",
    "Y NADIE SE HA PREGUNTADO LO QUE LE VA A COSTAR AL HOMBRE DEL GUANTAZO LA SITUACION..?!? PORQUE SEGURO ES, QUE EL MENDA MUSULMONO LE VA A PONER UNA DENUNCIA, QUE EL FALLO VA A SER "
    """CULPABLE"""
    ", QUE UNA PANDILLA DE MUSULMONOS LE VA A ESTAR ESPERANDO DELANTE DE LA PUERTA DE SU NEGOCIO https://t.co/DjfA63A0T2",
    "@Fed_Durand Callate come sobra, más zorra son las tuyas",
]

sentences = ["Easyjet quiere duplicar el número de mujeres piloto' Verás tú para aparcar el avión.", 
             "El gobierno debe crear un control estricto de inmigración en las zonas fronterizas con Colombia por q después del 20-8querrán venir en masa"
             ]


def remove_accent(text):
    return unidecode(text)


def tokenize_nltk(text):
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text, language="spanish")
    return tokens


def main():
    nlp = spacy.load("es_core_news_sm")
    for text in sentences:
        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        print(lemmatized_tokens)
        print(" ".join(lemmatized_tokens))
        # print(tokenize_nltk(text))


main()
