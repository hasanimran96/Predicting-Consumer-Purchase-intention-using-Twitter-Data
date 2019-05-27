# imports
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


def data_preprocessing(final_data_frame):
    # -----------------------------------------------------------------------
    # Data cleaning step wise
    # -----------------------------------------------------------------------
    # 1. LOWERCASE
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x.lower() for x in x.split())
    )
    # 2. REMOVE PUNC
    final_data_frame["text"] = final_data_frame["text"].str.replace(
        "[^\w\s]", "")
    # 3. STOPWORDS REMOVAL
    stop = stopwords.words("english")
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )
    # 4. COMMON WORD REMOVAL
    freq = pd.Series(
        " ".join(final_data_frame["text"]).split()).value_counts()[:10]
    freq = list(freq.index)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq)
    )
    # 5. RARE WORDS REMOVAL
    rare = pd.Series(
        " ".join(final_data_frame["text"]).split()).value_counts()[-10:]
    rare = list(rare.index)
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in rare)
    )
    # 6. SPELLING CORRECTION
    final_data_frame["text"][:5].apply(lambda x: str(TextBlob(x).correct()))
    # 7. STEMMING
    st = PorterStemmer()
    final_data_frame["text"][:5].apply(
        lambda x: " ".join([st.stem(word) for word in x.split()])
    )
    # 8. LEMMATIZATION
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
    )
    # -----------------------------------------------------------------------
    # BUILDING THE CORPUS
    # -----------------------------------------------------------------------
    corpus = []
    for text in final_data_frame["text"]:
        corpus.append(text)
    # -----------------------------------------------------------------------
    # CHANGE CLASS VALUES FROM YES/NO TO 0/1
    # -----------------------------------------------------------------------
    final_data_frame.rename(columns={"class": "class_label"}, inplace=True)
    Class_Label = {"yes": 1, "no": 0}
    final_data_frame.class_label = [
        Class_Label[item] for item in final_data_frame.class_label
    ]
    final_data_frame.rename(columns={"class_label": "class"}, inplace=True)
    return final_data_frame, corpus


def output_to_results(pathData_train, pathData_test, doc_vector, model):
    train_data, train_data_undefined = extract(pathData_train)
    test_data, test_data_undefined = extract(pathData_test)

    train_data, train_corpus = data_preprocessing(train_data)
    test_data, test_corpus = data_preprocessing(test_data)
    print(train_data.head())
    print(test_data.head())
    # -----------------------------------------------------------------------
    # Select Document vector
    # -----------------------------------------------------------------------
    if doc_vector == "TF":
        count_vectorizer = CountVectorizer()
        count_vectorized_data = count_vectorizer.fit_transform(train_corpus)
        vectorized_data_train = count_vectorized_data
        count_vectorizer = CountVectorizer()
        count_vectorized_data = count_vectorizer.fit_transform(test_corpus)
        vectorized_data_test = count_vectorized_data
    elif doc_vector == "IDF":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorized_data = tfidf_vectorizer.fit_transform(train_corpus)
        vectorized_data_train = tfidf_vectorized_data
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorized_data = tfidf_vectorizer.fit_transform(test_corpus)
        vectorized_data_test = tfidf_vectorized_data
    # -----------------------------------------------------------------------
    # training/testing
    # -----------------------------------------------------------------------
    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(
        vectorized_data_train, train_data["class"], test_size=0, random_state=0
    )
    X_train = X_train_temp
    Y_train = Y_train_temp
    X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(
        vectorized_data_test, test_data["class"], test_size=1, random_state=0
    )
    X_test = X_test_temp
    Y_test = Y_test_temp
    # X_train = vectorized_data_train
    # X_test = train_data["class"]
    # Y_train = vectorized_data_test
    # Y_test = test_data["class"]
    # -----------------------------------------------------------------------
    # Select model to train and display stats
    # -----------------------------------------------------------------------
    if model == "SVM":
        SVM = svm.SVC(probability=True, C=1.0,
                      kernel="linear", degree=3, gamma="auto")
        SVM.fit(X_train, Y_train)
        stats = report_results(SVM, X_test, Y_test)
    elif model == "naive":
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(X_train, Y_train)
        stats = report_results(Naive, X_test, Y_test)
    elif model == "logistic":
        logisticReg = linear_model.LogisticRegression(C=1.0)
        logisticReg.fit(X_train, Y_train)
        stats = report_results(logisticReg, X_test, Y_test)
    elif model == "DTC":
        DTC = DecisionTreeClassifier(min_samples_split=7, random_state=252)
        DTC.fit(X_train, Y_train)
        stats = report_results(DTC, X_test, Y_test)
    elif model == "neural":
        neural_network = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        neural_network.fit(X_train, Y_train)
        stats = report_results(neural_network, X_test, Y_test)
    # -----------------------------------------------------------------------
    return stats


result = output_to_results("data/training.csv", "data/test.csv", "IDF", "SVM")
print(result)
