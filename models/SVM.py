%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import NaiveBayes as nb

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)

# ------------------------------------------------------------------
path = "/home/hasan/Desktop/FYP-II/data/AnnotatedData3.csv"


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


final_data_frame, data_frame_undefined = extract(path)

final_data_frame.head()

# ---------------------------------------------------------------------------------
DATA_CLEAN = nb.NaiveBayesModel()
li_clean_text = DATA_CLEAN.clean_data(final_data_frame)
uniqueWords = DATA_CLEAN.make_unique_li(li_clean_text)

# --------------------------------------------------------------------------------
# data_clean = final_data_frame.copy()
# data_clean["sentiment"] = data_clean["class"].apply(lambda x: 1 if x == "no" else 0)
# data_clean["text_clean"] = data_clean["text"].apply(
#     lambda x: BeautifulSoup(x, "lxml").text
# )
# data_clean = data_clean.loc[:, ["text_clean", "sentiment"]]
# data_clean.head()

# ------------------------------------------------------------------------------
train, test = train_test_split(uniqueWords, test_size=0.2, random_state=1)
X_train = train["text"].values
X_test = test["text"].values
y_train = train["class"]
y_test = test["class"]

# -------------------------------------------------------------------------------
# def tokenize(text):
#     tknzr = TweetTokenizer()
#     return tknzr.tokenize(text)


# def stem(doc):
#     return (stemmer.stem(w) for w in analyzer(doc))


# # en_stopwords = set(stopwords.words("english"))
# en_stopwords = open("/home/hasan/Desktop/FYP-II/stopwords.txt", "r")
# stopwords = en_stopwords.readlines()
# li_stopwords = stopwords[0].split()
# li_stopwords = set(li_stopwords)
# en_stopwords = li_stopwords
# print(en_stopwords)

# vectorizer = CountVectorizer(
#     analyzer="word",
#     tokenizer=tokenize,
#     lowercase=True,
#     ngram_range=(2, 2),
#     stop_words=en_stopwords,
# )

# vectorizer = DATA_CLEAN.binary_docvector(final_data_frame, uniqueWords)


# -------------------------------------------------------------------------------
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

pipeline_svm = make_pipeline(
    SVC(probability=True, kernel="linear", class_weight="balanced")
)

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid={"svc__C": [0.01, 0.1, 1]},
    cv=kfolds,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)

grid_svm.best_params_
grid_svm.best_score_


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {"auc": auc, "f1": f1, "acc": acc, "precision": prec, "recall": rec}
    return result


report_results(grid_svm.best_estimator_, X_test, y_test)

# ----------------------------------------------------------------------------------
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)

fpr, tpr = roc_svm
plt.figure(figsize=(14, 8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc curve")
plt.show()

# ----------------------------------------------------------------------------------
prediction_result = grid_svm.predict(["i dont want to buy the iphone x"])

print("purchase intention of tweet is: ")

if prediction_result == 1:
    print("no")
else:
    print("yes")

