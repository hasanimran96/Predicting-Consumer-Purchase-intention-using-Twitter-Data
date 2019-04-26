# from pathConfig import PATH_CONFIG  # path config file imported
# from pages.pathConfig import PATH_CONFIG
# import pathConfig
import pandas as pd
import os
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)



# pathData = "uploadeddata\Annotated4.csv" # ubunutu/linux
# pathData = PATH_CONFIG['pathData'] #windows


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
    return result



def output_to_results(pathData):
    final_data_frame, data_frame_undefined = extract(pathData)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x.lower() for x in x.split())
    )
    final_data_frame["text"] = final_data_frame["text"].str.replace("[^\w\s]", "")
    stop = stopwords.words("english")
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )
    freq = pd.Series(" ".join(final_data_frame["text"]).split()).value_counts()[:10]
    freq = list(freq.index)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq)
    )
    rare = pd.Series(" ".join(final_data_frame["text"]).split()).value_counts()[-10:]
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
    SVM = svm.SVC(probability=True, C=1.0, kernel="linear", degree=3, gamma="auto")
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


def hello_world():
    return "Muhammad Ammar Saleem"

def hello():
    return hello_world()

def read_dir():
    
    path = "uploadeddata\\"
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        files.append(f)
    return files

print(output_to_results("uploadeddata/AnnotatedData3.csv"))