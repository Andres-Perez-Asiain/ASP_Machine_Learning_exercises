import glob
import nltk
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from string import digits, punctuation
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage, dendrogram

"""2. Speeches I"""
# a) Read up about pathlib.Path().glob(). Use it to read the text files in
# ./data/speeches into a corpus # (i.e. a list of strings), but only those that
# start with ”R0”. The files represent a non-random selection of # speeches of
# central bankers, which have already been stripped off meta information.
# The files are encoded # in UTF8, except for two broken ones. Use a
# try-except-statement to skip reading them and print the filename instead.

corpus = []
for file in glob.glob("./data/speeches/R0*"):
    try:
        new = open(file, encoding="utf-8").read()
    except:
        print(file)
        corpus.append(new)

# b)Vectorize the speeches using tfidf using 1-grams, 2-grams and 3-grams while
# removing English stopwords # and proper tokenization (i.e., you create a
# tfidf matrix).

_stemmer = nltk.snowball.SnowballStemmer("english")

def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and punctuation."""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]

 _stopwords = nltk.corpus.stopwords.words("english")
_stopwords = tokenize_and_stem(' '.join(_stopwords))

count = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
count.fit(corpus)
count_matrix = count.transform(corpus)
count.get_feature_names()
df_count =pd.DataFrame(count_matrix.todense().T,
                       index=count.get_feature_names())

tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words=_stopwords,
                        tokenizer=tokenize_and_stem)
tfidf_matrix = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(count_matrix.todense().T,
                       index=count.get_feature_names())

#c) Pickle the resulting sparse matrix using pickle.dump() as
# ./output/speech_matrix.pk. Save the terms as well as ./output/terms.csv

file = "./output/speech_matrix.pk"
output_file = open(file, 'wb')
pickle.dump(tfidf_matrix, output_file)
df_tfidf.to_csv("./output/terms.csv")

"""3. Speeches II"""
#a) Read the count-matrix from exercise ”Speeches I”
# (./output/speech_matrix.pk) using pickle.load().

file = "./output/speech_matrix.pk"
input_file = open(file, 'rb')
tfidf_matrix = pickle.load(input_file)
#input_file.close()

#b) Using the matrix, create a dendrogram of hierarchical clustering using the
#cosine distance and the complete linkage method. Remove the x-ticks from the
#plot. Optionally, set the color threshold such that three # clusters are shown

agg = AgglomerativeClustering(n_clusters=3, affinity="cosine",
                              linkage="complete")
agg.fit(tfidf_matrix.todense())

cluster = pd.DataFrame(tfidf_matrix.todense())
cluster["aggclustering"] = agg.labels_

# Dendogram
linkage_matrix = linkage(tfidf_matrix.todense(), metric="cosine",
                         method="complete")
dendrogram(linkage_matrix)
plt.tick_params(axis="x", bottom=False)
plt.show()

#c) Save the dendrogram as ./output/speeches_dendrogram.pdf.
plt.savefig("./output/speeches_dendrogram.pdf")

"""4. Job Ads"""
#a) Read the text file ./data/Stellenanzeigen.txt and parse the lines such that
# you obtain a pandas.DataFrame()with three columns: ”Newspaper”, ”Date”,
# ”Job Ad”. Make sure to set the Date column as datetime type.

text_file = open("./data/Stellenanzeigen.txt", 'r', encoding="utf8")
lines = text_file.read().split("\n\n")
df = pd.DataFrame(lines)

new = df[0].str.split(",", n=1, expand=True)
df["Newspaper"] = new[0]

new = new[1].str.split("\n", n=1, expand=True)
df["Date"] = new[0]
df["Job Ad"] = new[1]
df = df.drop(columns=[0])

df["Date"] = df["Date"].str.replace(".", "")
df["Date"] = df["Date"].str.replace(" ", "/")
df["Date"] = df["Date"].str.replace("März", "03")
df["Date"] = pd.to_datetime(df["Date"], format="/%d/%m/%Y")

#b) Create a new column counting the number of words per job ad.
# Plot the average job ad length by decade.

df["length"] = [len(ad.split()) for ad in df["Job Ad"]]
df["years"] = df["Date"].dt.year

decades = []
for year in df["years"]:
    decade = int(np.floor(year / 10) * 10)
    decades.append(decade)

df["decades"] = decades
Avg_decade = df["length"].groupby(df.decades).mean().rename("Avg_length")\
    .reset_index()

sns.relplot(data=Avg_decade, x="decades", y="Avg_length")