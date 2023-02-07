from icecream import ic as ic
import pandas as pd
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from ru_number_to_text.num2t4ru import num2text
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['который', 'это', 'из-за', 'котором', 'который'])

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
mystem = Mystem()
df = pd.read_csv('input_0.csv')


def del_spaces(text):
    tmp = len(text)+1
    while (tmp>len(text)):
        tmp = len(text)
        text = text.replace("  ", ' ')
    return text

def digit_to_words(word):
    if word.isdigit():
        return num2text(int(word))
    return word


def preprocess_text(text):
    text = " ".join([digit_to_words(word) for word in text.split(' ') if not '@' in word])
    tokens = mystem.lemmatize(text)
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return del_spaces("".join([c for c in text if c.isalpha() or c ==' '])).strip()

def clean_tweets(df):
    #set up punctuations we want to be replaced
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    tempArr = []
    for line in df:
        # send to tweet_processor

        # замена аббревиатур
        line = replace_abbr(line, 'adict.csv')
        ic(line)
        tmpL = preprocess_text(line)
        # print(line)
        # print(tmpL)
        # remove puctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases

        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr

def replace_abbr(text, adict=None):
    regex = r"[А-ЯA-Z]+[А-ЯA-Z]"
    abbr = pd.read_csv(adict)

    matches = re.finditer(regex, text, re.MULTILINE)

    for matchNum, match in enumerate(matches, start=1):

        i=abbr.loc[abbr["short"]==match.group()]
        # .values[0][1]
        if len(i)>0:
            text = text.replace(match.group(), i.values[0][1])

        # text.replace(match, )
    return text

for i in df.index:
    df.at[i, 'post_text'] = df.at[i, 'post_text'].replace('\n', ' ')


df['clean tweet'] = clean_tweets(df['post_text'])
df.to_csv('test1.csv')
corpus = list(df['clean tweet'])
# ic(corpus)

corpus_embeddings = embedder.encode(corpus)
# vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.95, min_df=0.1)
# corpus_embeddings = vectorizer.fit_transform(corpus)

# print(corpus_embeddings)

num_clusters = 11
clustering_model = KMeans(n_clusters=num_clusters, max_iter=1000, random_state=0, tol=1e-3)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

cluster_df = pd.DataFrame(corpus, columns = ['corpus'])
cluster_df['cluster'] = cluster_assignment
cluster_df['readable']=df['post_text']

cluster_df.to_csv('res+.csv')

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1, "length ", len(cluster))
    print(cluster)
    print("")


def word_cloud(pred_df, label):
    wc = ' '.join([text for text in pred_df['corpus'][pred_df['cluster'] == label]])
    wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).generate(wc)
    fig7 = plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')
    fig7.show()

# num_clusters=num_clusters
# for i in range(num_clusters):
#     word_cloud(cluster_df,i)
