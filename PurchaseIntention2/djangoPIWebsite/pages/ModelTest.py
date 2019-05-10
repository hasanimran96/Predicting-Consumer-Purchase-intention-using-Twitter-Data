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
    # print(X.shape)
    # print(y.shape)
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
    return result, pred, pred_proba


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


def output_to_results(pathData_train, pathData_test, doc_vector, model, l1, l2, l3):
    train_data, train_data_undefined = extract(pathData_train)
    test_data, test_data_undefined = extract(pathData_test)
    # temp_text_sr = test_data['text']
    temp_test_data = test_data.copy(deep=True)
    train_data, train_corpus = data_preprocessing(train_data)
    test_data, test_corpus = data_preprocessing(test_data)
    # -----------------------------------------------------------------------
    # Select Document vector
    # -----------------------------------------------------------------------
    if doc_vector == "TF":
        count_vectorizer = CountVectorizer()
        count_vectorized_data_train = count_vectorizer.fit_transform(train_corpus)
        vectorized_data_train = count_vectorized_data_train
        # count_vectorized_data_test = count_vectorizer.fit_transform(test_corpus)
        count_vectorized_data_test = count_vectorizer.transform(test_corpus)
        vectorized_data_test = count_vectorized_data_test
    elif doc_vector == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorized_data_train = tfidf_vectorizer.fit_transform(train_corpus)
        vectorized_data_train = tfidf_vectorized_data_train
        tfidf_vectorized_data_test = tfidf_vectorizer.transform(test_corpus)
        vectorized_data_test = tfidf_vectorized_data_test
    # -----------------------------------------------------------------------
    # training/testing
    # -----------------------------------------------------------------------
    X_train = vectorized_data_train
    X_test = vectorized_data_test
    Y_train = train_data["class"].values
    Y_test = test_data["class"].values    
    # -----------------------------------------------------------------------
    # Select model to train and display stats
    # -----------------------------------------------------------------------
    if model == "SVM":
        SVM = svm.SVC(probability=True, C=1.0,
                      kernel="linear", degree=3, gamma="auto")
        SVM.fit(X_train, Y_train)
        stats, pred, pred_proba = report_results(SVM, X_test, Y_test)
    elif model == "Naive Bayes":
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(X_train, Y_train)
        stats, pred, pred_proba = report_results(Naive, X_test, Y_test)
    elif model == "Logistic Regression":
        logisticReg = linear_model.LogisticRegression(C=1.0)
        logisticReg.fit(X_train, Y_train)
        stats, pred, pred_proba = report_results(logisticReg, X_test, Y_test)
    elif model == "Decision Tree":
        DTC = DecisionTreeClassifier(min_samples_split=7, random_state=252)
        DTC.fit(X_train, Y_train)
        stats, pred, pred_proba = report_results(DTC, X_test, Y_test)
    elif model == "Neural Network":
        neural_network = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        neural_network.fit(X_train, Y_train)
        stats, pred, pred_proba= report_results(neural_network, X_test, Y_test)
        
    # -----------------------------------------------------------------------
    test_data['Predicted Class'] = pred.tolist()
    test_data['score'] = pred_proba.tolist()
    # print(test_data['score'])
    test_data = test_data.drop(['text'], axis=1)
    # print(test_data)
    
    test_data['class'].replace(0,"no",inplace=True)
    test_data['class'].replace(1,"yes",inplace=True)
    test_data['Predicted Class'].replace(0,"no",inplace=True)
    test_data['Predicted Class'].replace(1,"yes",inplace=True)
    test_data['text'] = temp_test_data['text'].tolist()
    test_data = test_data[['text', 'class', 'Predicted Class', 'score']]
    # print( test_data)
    potential_yes =  test_data["Predicted Class"] == "yes"
    potential_df = test_data[potential_yes]
    # print(test_data['Predicted Class'])
    # print(potential_df['score'])
    level=[]
    for score in potential_df['score']:
        if score >= int(l1)/100:
            level.append("1")
        elif score >= int(l2)/100 and score < int(l1)/100:
            level.append("2") 
        elif score < int(l2)/100:
            level.append("3") 
    # print(level)          
    potential_df['Levels'] = level 
    potential_df.sort_values(by=['score'], inplace=True, ascending=False)
    pie_data = potential_df['Levels'].value_counts()         
    return stats, test_data, potential_df, pie_data


# result = output_to_results("uploadeddata\Annotated4.csv",
#                            "uploadeddata\AnnotatedData2.csv", "TF-IDF", "Naive Bayes","90","80","70")
# # print(result)
