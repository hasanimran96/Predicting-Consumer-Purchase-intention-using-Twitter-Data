import pathConfig as pc  # path config file imported
import pandas as pd

pathData = "data/AnnotatedData3.csv"  # ubunutu/linux
# pathData = pc.PATH_CONFIG['pathData'] #windows


def extract(path):
    fd = open(path, encoding="utf-8", errors="replace")
    df = pd.read_csv(fd)
    defined = df["class"] != ("undefined")
    # #output dataframe without undeined
    df2 = df[defined]
    defined1 = df2["class"] != "Undefined"
    df4 = df2[defined1]
    # replace no PI with no
    df3 = df4.replace("No PI", "no")
    # replace PI with yes
    final = df3.replace("PI", "yes")

    replace_yes = final.replace("Yes", "yes")
    final_df = replace_yes.replace("No", "no")
    return final_df, df


final_data_frame, data_frame_undefined = extract(pathData)

print(final_data_frame.head())
print()

# ---------------------------------------------------------------------
# LOWERCASE
final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join(x.lower() for x in x.split())
)
print("lowercase all text")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# REMOVE PUNC
final_data_frame["text"] = final_data_frame["text"].str.replace("[^\w\s]", "")
print("removed punctuation")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# STOPWORDS REMOVAL
from nltk.corpus import stopwords

stop = stopwords.words("english")
final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop)
)
print("removed stoped words")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# COMMON WORD REMOVAL
freq = pd.Series(" ".join(final_data_frame["text"]).split()).value_counts()[:10]
freq = list(freq.index)
final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join(x for x in x.split() if x not in freq)
)
print("removed comman words")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# RARE WORDS REMOVAL
rare = pd.Series(" ".join(final_data_frame["text"]).split()).value_counts()[-10:]
rare = list(rare.index)
final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join(x for x in x.split() if x not in rare)
)
print("removed rare words")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# SPELLING CORRECTION
from textblob import TextBlob

final_data_frame["text"][:5].apply(lambda x: str(TextBlob(x).correct()))
print("fixed spellings")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# STEMMING
from nltk.stem import PorterStemmer

st = PorterStemmer()
final_data_frame["text"][:5].apply(
    lambda x: " ".join([st.stem(word) for word in x.split()])
)
print("applied stemming")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# LEMMATIZATION
# Lemmatization is a more effective option than stemming
# because it converts the word into its root word,
# rather than just stripping the suffices.
from textblob import Word

final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
)
print("applied lemmatization")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

