from features.preprocess import Data

lines = [
    "Me estoy comiendo la picada árabe más rica de mi vida",
    "Estoy haciendo el check-in a una muchacha de los Emiratos Árabes. Ella: ese tatuaje es árabe, ¿no?  Yo: sí 😨 (yo pensando pa dentro: por Dios y por Alá que el significado sea lo que yo quería 😰) Ella: es tal (una palabra rara en árabe) Yo: 🤨 Me la dice en inglés y yo: ¡Sí! 😅",
    "MMMMM TU NADA MÁS KERIAS ENSEÑAR LAS CHCHIS PINCHE BIEJA PUTA!!!!",
    "¡Cállate maldita escoria! ¡¿Acaso no sabes con quién hablas?! ¡Hablas con Baron explodo kills el mejor héroe de U.A quien destrozara tu jodida cara de mierda! ¡Haré que te arrepientas y te metas tus asquerosas idioteces por el orto! https://t.co/lxQnHmqZfd",
]

sentence = "gtfo #gtfoo gtfomy country"


def main():
    data = Data("", "", "es")
    for line in lines:
        processed_line = data._process_line(line)
        print(processed_line)


main()
