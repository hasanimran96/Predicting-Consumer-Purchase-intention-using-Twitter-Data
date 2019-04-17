import NaiveBayes as nb
import EvaluateModel as em
import NaiveBayesTextBlob as tb
import pandas as pd
import Clean as cn
import docVector as dv
import pathConfig as pc


#--------------------------------
###Path
pathStopwords = pc.PATH_CONFIG['pathStopWords']
pathData = pc.PATH_CONFIG['pathData']

def naive_bayes():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    df_clean, uniqueWords = clean.Clean()
    # print(uniqueWords)
    df_clean_test, df_clean_train = split(df_clean, 0, int(.3*(df_clean['class'].count())))
    print("test data count ",df_clean_test['class'].count())
    print("train data count ", df_clean_train['class'].count())
    docVector = doc_vector.DocVector(df_clean_train, uniqueWords)
    df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
    predict_df, test_data = model.predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,df_clean_test,clean)

    print("--------------Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ",stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ",stats.Precision(TP, FP))
    print("Recall = ",stats.Recall(TP, FN))
    print("fScore = ",stats.fScore(TP, FN, FP))
    print("True Negative = ", stats.TrueNegative(TN, FP))
    print("---------------------------------------------------------------------")


def text_blob():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    df_clean, uniqueWords = clean.Clean()
    df_clean_test, df_clean_train = split(df_clean, 0, int(.3 * (df_clean['class'].count())))
    docVector = doc_vector.DocVector(df_clean_train, uniqueWords)

    polarity_docVector = tb.text_blob(docVector, uniqueWords)
    # print(polarity_docVector['bad'])
    df_WordGivenPI, df_WordGivenNoPi, Prob_PI, Prob_NoPI, numWordsInPI, numWordsInNoPI = model.TrainModel(polarity_docVector, uniqueWords)
    predict_df, test_data = model.predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,df_clean_test,clean)

    print("--------------Naive Bayes with Text Blob Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ", stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ", stats.Precision(TP, FP))
    print("Recall = ", stats.Recall(TP, FN))
    print("fScore = ", stats.fScore(TP, FN, FP))
    print("True Negative = ", stats.TrueNegative(TN, FP))
    print("---------------------------------------------------------------------")

def binary_naive_bayes():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    df_clean, uniqueWords = clean.Clean()
    df_clean_test, df_clean_train = split(df_clean, 0, int(.3 * (df_clean['class'].count())))
    docVector = doc_vector.binary_docvector(df_clean_train, uniqueWords)
    # print(docVector)
    df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
    # print("Model Trained")
    predict_df, test_data = model.predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,df_clean_test,clean)

    print("--------------Binary Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ",stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ",stats.Precision(TP, FP))
    print("Recall = ",stats.Recall(TP, FN))
    print("fScore = ",stats.fScore(TP, FN, FP))
    print("True Negative = ", stats.TrueNegative(TN, FP))
    print("---------------------------------------------------------------------")

def split(df_final, start, end):
    df_test, df_train1, df_train2 = df_final.iloc[start:end,:], df_final.iloc[end:, :],df_final.iloc[0:start,:]
    df_train = pd.concat([df_train1, df_train2])
    return df_test, df_train


def Average(lst):
    return sum(lst) / len(lst)


def binary_naive_bayes_kfold():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    final_df, df = clean.extract(pathData)
    count = 0
    start = -200
    end = 0
    accuracy = []
    precision = []
    recall = []
    fscore = []
    true_neg = []
    stats = em.Evaluate()
    for count in range(5):
        start = start+200
        end = end + 200
        df_test, df_train = split(final_df, start, end)
        # print(df_train)
        li_clean_text, df_clean = clean.clean_data(df_train)
        uniqueWords = clean.make_unique_li(li_clean_text)
    # # print(uniqueWords)
        docVector = doc_vector.binary_docvector(df_clean, uniqueWords)
        df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
        predict_df, punc_df = model.predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI, df_test,clean)
        # print("--------------Naive Bayes Accuracy Stats---------------------------")
        TP, FN, TN, FP = stats.confusion_matrix(punc_df, predict_df)
        accuracy.append(stats.Accuracy(TP, TN, FP, FN))
        precision.append(stats.Precision(TP, FP))
        recall.append(stats.Recall(TP, FN))
        fscore.append(stats.fScore(TP, FN, FP))
        true_neg.append(stats.TrueNegative(TN,FP))
        # print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    print("Binary Naive Bayes wit k-fold Accuracy Stats")
    print("accuracy = ",accuracy)
    print("precison = ", precision)
    print("recall = ", recall)
    print("f-score = ", fscore)
    print("True Negative = ",true_neg)
    print("accuracy = ", Average(accuracy))
    print("precison = ", Average(precision))
    print("recall = ", Average(recall))
    print("f-score = ", Average(fscore))
    print("true negative = ",Average(true_neg))

def tf_idf_naive_bayes():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    df_clean, uniqueWords = clean.Clean()
    df_clean_test, df_clean_train = split(df_clean, 0, int(.3 * (df_clean['class'].count())))
    docVector = doc_vector.tf_idf(df_clean_train,uniqueWords)
    # print(docVector)
    df_WordGivenPI, df_WordGivenNoPi, Prob_PI, Prob_NoPI, numWordsInPI, numWordsInNoPI = model.TrainModel(docVector,
                                                                                                          uniqueWords)
    # print("Model Trained")
    predict_df, test_data = model.predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi,
                                          numWordsInPI, numWordsInNoPI,df_clean_test, clean)

    print("--------------tf-idf Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ", stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ", stats.Precision(TP, FP))
    print("Recall = ", stats.Recall(TP, FN))
    print("fScore = ", stats.fScore(TP, FN, FP))
    print("True Negative = ", stats.TrueNegative(TN, FP))
    print("---------------------------------------------------------------------")
    # print("Saad Saad Saad")

binary_naive_bayes_kfold()
text_blob()
naive_bayes()
binary_naive_bayes()
tf_idf_naive_bayes()