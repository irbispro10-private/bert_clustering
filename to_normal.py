import nltk, csv
nltk.download("stopwords")
from string import punctuation
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['который', 'это', 'из-за', 'котором', 'который'])
from pymystem3 import Mystem
mystem = Mystem()


def del_spaces(text):
    tmp = len(text)+1
    while (tmp>len(text)):
        tmp = len(text)
        text = text.replace("  ", ' ')
    return text

def preprocess_text(text):
    tokens = mystem.lemmatize(text)
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return del_spaces("".join([c for c in text if c.isalpha() or c ==' '])).strip()

with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(preprocess_text(row[0]))

