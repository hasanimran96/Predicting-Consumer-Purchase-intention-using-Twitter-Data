import pandas as pd
import numpy as np
import nltk
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def text_concat():
    text = ""
    for x in final_df["text"]:
        text = text + str(x)
    return text


def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w.lower())
    # return list of corpus without stop words in a list.
    return filtered_sentence


def removePunc(eachText):
    remove_punc = re.sub(r'[^\w\s]', '', eachText)
    # return corpus with out punctuation.
    return remove_punc


def clean_data(corpus):
    remove_punc = removePunc(corpus)
    li_remove_stopWords = removeStopWords(remove_punc)
    return li_remove_stopWords


def make_unique_li(li_cleanText):
    unique_words_set = set(li_cleanText)
    unique_word_li = list(unique_words_set)
    return unique_word_li


df = pd.read_csv('D:/DATA/Sem8/fyp/AnnotatedData.csv')
# create series of true and false. True is assigned for yes/no. False for undefine
defined = df['Purchase Intention'] != "undefined"
# output dataframe without undeined
df2 = df[defined]
# replace no PI with no
df3 = df2.replace("No PI", "no")
# replace PI with yes
final_df = df3.replace("PI", "yes")

# create series of true and false. False is assigned for yes/no. True is for undefined
undefined = df['Purchase Intention'] == "undefined"
# output dataframe without undeined
df3 = df[undefined]

corpus = text_concat()

li_cleaned_text = clean_data(corpus)
# print(li_cleaned_text)

uniqueWords = print(make_unique_li(li_cleaned_text))
