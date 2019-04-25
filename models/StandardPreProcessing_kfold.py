from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
from nltk.stem import PorterStemmer
from textblob import TextBlob
from nltk.corpus import stopwords
from pathConfig import PATH_CONFIG  # path config file imported
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


pathData = "data/AnnotatedData3.csv"  # ubunutu/linux
# pathData = PATH_CONFIG['pathData'] #windows


def handle_negation(final_df):
    out_df = pd.DataFrame()
    count_tweet = 0
    for text in final_df['text']:
        temp_text = ""
        li_text = text.split()
        for word in li_text:
            count = 0
            lower_word = word.lower()
            if lower_word == "didn't" or lower_word == "not" or lower_word == "no" or lower_word == "never"\
                    or lower_word == "don't":
                temp = count + 1
                temp_text = temp_text + word + " "
                for i in range(temp, len(li_text)):
                    if li_text[i] in [",", "?", "!", "."]:
                        temp_text = " "+temp_text + li_text[i] + " "
                        break
                    else:
                        temp_text = temp_text + "NOT_" + li_text[i]+" "

            else:
                temp_text = temp_text + word + " "
        # print(temp_text)
        out_df.at[count_tweet, 'text'] = temp_text
        out_df.at[count_tweet, 'class'] = final_df.iloc[count_tweet]['class']
        count_tweet += 1
    return out_df


def space(final_df):
    new_df = pd.DataFrame()
    count_tweets = 0
    for text in final_df['text']:
        temp = ""
        for char in text:
            if char in [",", ".", "!", "?", ":", ";"]:
                temp = temp + ' ' + char

            else:
                temp = temp + char
        # print(temp)
        new_df.at[count_tweets, 'text'] = temp
        new_df.at[count_tweets, 'class'] = final_df.iloc[count_tweets]['class']
        count_tweets += 1
    # print("new_df")
    # print(new_df)
    return new_df


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


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    TrueNeg = tn / (tn + fp)
    result = {
        "auc": auc,
        "f1": f1,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "True Negative rate": TrueNeg,
    }
    return acc, TrueNeg, prec, rec


final_data_frame, data_frame_undefined = extract(pathData)

print(final_data_frame.head())
print()

# ---------------------------------------------------------------------
# SHUFFLING THE DATA FRAME
final_data_frame.reindex(np.random.permutation(final_data_frame.index))
print("shuffled data frame")
print(final_data_frame.head())
print()
# ---------------------------------------------------------------------

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
# NEGATION HANDLING
final_data_frame = space(final_data_frame)
final_data_frame = handle_negation(final_data_frame)
print("handle negation")
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
freq = pd.Series(
    " ".join(final_data_frame["text"]).split()).value_counts()[:2]
print(freq)
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
rare = pd.Series(
    " ".join(final_data_frame["text"]).split()).value_counts()[-10:]
print(rare)
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

final_data_frame["text"][:5].apply(lambda x: str(TextBlob(x).correct()))
print("fixed spellings")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# STEMMING

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

final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
)
print("applied lemmatization")
print(final_data_frame["text"].head())
print()
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDING THE CORPUS
corpus = []
for text in final_data_frame["text"]:
    corpus.append(text)
print("corpus")
# print(corpus)
print()
# ---------------------------------------------------------------------

# -------------------------------------------------------------------------
# CHANGE CLASS VALUES FROM YES/NO TO 0/1
final_data_frame.rename(columns={"class": "class_label"}, inplace=True)
Class_Label = {"yes": 1, "no": 0}
final_data_frame.class_label = [
    Class_Label[item] for item in final_data_frame.class_label
]
final_data_frame.rename(columns={"class_label": "class"}, inplace=True)
print("rename values of class column")
print(final_data_frame.head())
print()
# -------------------------------------------------------------------------

# --------------------------------------------------------------------------
# TF
# Transforms text into a sparse matrix of n-gram counts.

count_vectorizer = CountVectorizer()
count_vectorized_data = count_vectorizer.fit_transform(corpus)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# IDF
# Performs the TF-IDF transformation from a provided matrix of counts.

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorized_data = tfidf_vectorizer.fit_transform(corpus)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# chose document vector
# vectorized_data = count_vectorized_data
vectorized_data = tfidf_vectorized_data
# --------------------------------------------------------------------------

# # --------------------------------------------------------------------------
# # SPLITING THE DATA

# X_train, X_test, Y_train, Y_test = train_test_split(
#     vectorized_data, final_data_frame["class"], test_size=0.3, random_state=0
# )
# # --------------------------------------------------------------------------

acc_SVM = 0.0
TrueNeg_SVM = 0.0
prec_SVM = 0.0
recall_SVM = 0.0
acc_Naive = 0.0
TrueNeg_Naive = 0.0
prec_Naive = 0.0
recall_Naive = 0.0
acc_log = 0.0
TrueNeg_log = 0.0
prec_log = 0.0
recall_log = 0.0
acc_dtc = 0.0
TrueNeg_dtc = 0.0
prec_dtc = 0.0
recall_dtc = 0.0
acc_nn = 0.0
TrueNeg_nn = 0.0
prec_nn = 0.0
recall_nn = 0.0

# --------------------------------------------------------------------------
# APPLYING KFOLD CROSS VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=None)

for train_index, test_index in kf.split(final_data_frame):
    #print("Train:", train_index, "Validation:", test_index)
    X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
    Y_train, Y_test = final_data_frame["class"][train_index], final_data_frame["class"][test_index]
# --------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Applying SVM

    SVM = svm.SVC(probability=True, C=1.0,
                  kernel="linear", degree=3, gamma="auto")
    SVM.fit(X_train, Y_train)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Applying Naive Bayes

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(X_train, Y_train)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Applying Logistic Regression

    logisticReg = linear_model.LogisticRegression(C=1.0)
    logisticReg.fit(X_train, Y_train)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Applying Decision Tree

    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=252)
    dtc.fit(X_train, Y_train)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Applying Neural Network

    neural_network = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(15, 15, 5), random_state=1
    )
    neural_network.fit(X_train, Y_train)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # AVERAGING THE ACCURACY OF THE MODELS
    temp_acc_SVM, temp_TrueNeg_SVM, temp_prec_SVM, temp_recall_SVM = report_results(
        SVM, X_test, Y_test)
    temp_acc_Naive, temp_TrueNeg_Naive, temp_prec_Naive, temp_recall_Naive = report_results(
        Naive, X_test, Y_test)
    temp_acc_log, temp_TrueNeg_log, temp_prec_log, temp_recall_log = report_results(
        logisticReg, X_test, Y_test)
    temp_acc_dtc, temp_TrueNeg_dtc, temp_prec_dtc, temp_recall_dtc = report_results(
        dtc, X_test, Y_test)
    temp_acc_nn, temp_TrueNeg_nn, temp_prec_nn, temp_recall_nn = report_results(
        neural_network, X_test, Y_test)

    acc_SVM = acc_SVM + temp_acc_SVM
    TrueNeg_SVM = TrueNeg_SVM + temp_TrueNeg_SVM
    prec_SVM = prec_SVM + temp_prec_SVM
    recall_SVM = recall_SVM + temp_recall_SVM
    acc_Naive = acc_Naive + temp_acc_Naive
    TrueNeg_Naive = TrueNeg_Naive + temp_TrueNeg_Naive
    prec_Naive = prec_Naive + temp_prec_Naive
    recall_Naive = recall_Naive + temp_recall_Naive
    acc_log = acc_log + temp_acc_log
    TrueNeg_log = TrueNeg_log + temp_TrueNeg_log
    prec_log = prec_log + temp_prec_log
    recall_log = recall_log + temp_recall_log
    acc_dtc = acc_dtc + temp_acc_dtc
    TrueNeg_dtc = TrueNeg_dtc + temp_TrueNeg_dtc
    prec_dtc = prec_dtc + temp_prec_dtc
    recall_dtc = recall_dtc + temp_recall_dtc
    acc_nn = acc_nn + temp_acc_nn
    TrueNeg_nn = TrueNeg_nn + temp_TrueNeg_nn
    prec_nn = prec_nn + temp_prec_nn
    recall_nn = recall_nn + temp_recall_nn

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


acc_SVM = acc_SVM / 5
TrueNeg_SVM = TrueNeg_SVM / 5
prec_SVM = prec_SVM / 5
recall_SVM = recall_SVM / 5
acc_Naive = acc_Naive / 5
TrueNeg_Naive = TrueNeg_Naive / 5
prec_Naive = prec_Naive / 5
recall_Naive = recall_Naive / 5
acc_log = acc_log / 5
TrueNeg_log = TrueNeg_log / 5
prec_log = prec_log / 5
recall_log = recall_log / 5
acc_dtc = acc_dtc / 5
TrueNeg_dtc = TrueNeg_dtc / 5
prec_dtc = prec_dtc / 5
recall_dtc = recall_dtc / 5
acc_nn = acc_nn / 5
TrueNeg_nn = TrueNeg_nn / 5
prec_nn = prec_nn / 5
recall_nn = recall_nn / 5

# -------------------------------------------------------------------------
# statitics for SVM
print("-------------------------------------------------------------------------")
print("statitics for SVM")
stats = {
    "acc": acc_SVM,
    "True Negative rate": TrueNeg_SVM,
    "precision": prec_SVM,
    "recall": recall_SVM,
}
print(stats)
print("-------------------------------------------------------------------------")
print()
# statitics for NaiveBayes
print("-------------------------------------------------------------------------")
print("statitics for NaiveBayes")
stats = {
    "acc": acc_Naive,
    "True Negative rate": TrueNeg_Naive,
    "precision": prec_Naive,
    "recall": recall_Naive,
}
print(stats)
print("-------------------------------------------------------------------------")
print()
# statitics for LogisticRegression
print("-------------------------------------------------------------------------")
print("statitics for Logistic Regression")
stats = {
    "acc": acc_log,
    "True Negative rate": TrueNeg_log,
    "precision": prec_log,
    "recall": recall_log,
}
print(stats)
print("-------------------------------------------------------------------------")
print()
# statitics for DECISION TREE
print("-------------------------------------------------------------------------")
print("statitics for decision tree")
stats = {
    "acc": acc_dtc,
    "True Negative rate": TrueNeg_dtc,
    "precision": prec_dtc,
    "recall": recall_dtc,
}
print(stats)
print("-------------------------------------------------------------------------")
print()
# statistics for neural network
print("-------------------------------------------------------------------------")
print("statitics for Neural Network")
stats = {
    "acc": acc_nn,
    "True Negative rate": TrueNeg_nn,
    "precision": prec_nn,
    "recall": recall_nn,
}
print(stats)
print("-------------------------------------------------------------------------")
print("")
# --------------------------------------------------------------------------


def output_to_results(pathData):
    final_data_frame, data_frame_undefined = extract(pathData)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x.lower() for x in x.split())
    )
    final_data_frame["text"] = final_data_frame["text"].str.replace(
        "[^\w\s]", "")
    stop = stopwords.words("english")
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )
    freq = pd.Series(
        " ".join(final_data_frame["text"]).split()).value_counts()[:10]
    freq = list(freq.index)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq)
    )
    rare = pd.Series(
        " ".join(final_data_frame["text"]).split()).value_counts()[-10:]
    rare = list(rare.index)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in rare)
    )
    final_data_frame["text"][:5].apply(lambda x: str(TextBlob(x).correct()))
    st = PorterStemmer()
    final_data_frame["text"][:5].apply(
        lambda x: " ".join([st.stem(word) for word in x.split()])
    )
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
    )
    corpus = []
    for text in final_data_frame["text"]:
        corpus.append(text)
    final_data_frame.rename(columns={"class": "class_label"}, inplace=True)
    Class_Label = {"yes": 1, "no": 0}
    final_data_frame.class_label = [
        Class_Label[item] for item in final_data_frame.class_label
    ]
    final_data_frame.rename(columns={"class_label": "class"}, inplace=True)
    count_vectorizer = CountVectorizer()
    count_vectorized_data = count_vectorizer.fit_transform(corpus)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_data = tfidf_vectorizer.fit_transform(corpus)
    vectorized_data = tfidf_vectorized_data
    X_train, X_test, Y_train, Y_test = train_test_split(
        vectorized_data, final_data_frame["class"], test_size=0.3, random_state=0
    )
    SVM = svm.SVC(probability=True, C=1.0,
                  kernel="linear", degree=3, gamma="auto")
    SVM.fit(X_train, Y_train)
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(X_train, Y_train)
    logisticReg = linear_model.LogisticRegression(C=1.0)
    logisticReg.fit(X_train, Y_train)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=252)
    dtc.fit(X_train, Y_train)
    neural_network = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    )
    neural_network.fit(X_train, Y_train)
    stats_SVM = report_results(SVM, X_test, Y_test)
    stats_Naive = report_results(Naive, X_test, Y_test)
    stats_logistic = report_results(logisticReg, X_test, Y_test)
    stats_dtc = report_results(dtc, X_test, Y_test)
    stats_neural = report_results(neural_network, X_test, Y_test)
    stats = []
    stats.append(stats_SVM)
    stats.append(stats_Naive)
    stats.append(stats_logistic)
    stats.append(stats_dtc)
    stats.append(stats_neural)
    return stats


# output_to_results("data/AnnotatedData3.csv")
