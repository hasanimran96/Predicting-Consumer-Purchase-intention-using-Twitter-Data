import NaiveBayes as nb
import EvaluateModel as em
import NaiveBayesTextBlob as tb
import pandas as pd

def naive_bayes():
    model = nb.NaiveBayesModel()
    path = 'E:/DATA/Sem8/fyp/Training.csv'
    final_df, df = model.extract('E:/DATA/Sem8/fyp/Training.csv')
    # corpus = model.text_concat(final_df)
    li_clean_text = model.clean_data(final_df)
    uniqueWords = model.make_unique_li(li_clean_text)
    # print(uniqueWords)
    docVector = model.DocVector(final_df, uniqueWords)
    df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
    predict_df, test_data = model.Predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI)

    print("--------------Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ",stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ",stats.Precision(TP, FP))
    print("Recall = ",stats.Recall(TP, FN))
    print("fScore = ",stats.fScore(TP, FN, FP))
    print("---------------------------------------------------------------------")

def text_blob():
    model = nb.NaiveBayesModel()
    path = 'E:/DATA/Sem8/fyp/Training.csv'
    final_df, df = model.extract('E:/DATA/Sem8/fyp/Training.csv')
    corpus = model.text_concat(final_df)
    li_clean_text = model.clean_data(corpus)
    uniqueWords = model.make_unique_li(li_clean_text)
    docVector = model.DocVector(final_df, uniqueWords)
    polarity_docVector = tb.text_blob(docVector, uniqueWords)
    print(polarity_docVector['bad'])
    df_WordGivenPI, df_WordGivenNoPi, Prob_PI, Prob_NoPI, numWordsInPI, numWordsInNoPI = model.TrainModel(polarity_docVector, uniqueWords)
    predict_df, test_data = model.Predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI)

    print("--------------Naive Bayes with Text Blob Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ", stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ", stats.Precision(TP, FP))
    print("Recall = ", stats.Recall(TP, FN))
    print("fScore = ", stats.fScore(TP, FN, FP))
    print("---------------------------------------------------------------------")

def binary_naive_bayes():
    model = nb.NaiveBayesModel()
    path = 'E:/DATA/Sem8/fyp/Training.csv'
    final_df, df = model.extract('E:/DATA/Sem8/fyp/Training.csv')
    li_clean_text = model.clean_data(final_df)
    uniqueWords = model.make_unique_li(li_clean_text)
    # print(uniqueWords)
    docVector = model.binary_docvector(final_df, uniqueWords)
    df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
    predict_df, test_data = model.Predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI)

    print("--------------Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ",stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ",stats.Precision(TP, FP))
    print("Recall = ",stats.Recall(TP, FN))
    print("fScore = ",stats.fScore(TP, FN, FP))
    print("---------------------------------------------------------------------")

def split(df_final, start, end):
    df_test, df_train1, df_train2 = df_final.iloc[start:end,:], df_final.iloc[end:, :],df_final.iloc[0:start,:]
    df_train = pd.concat([df_train1, df_train2])
    return df_test, df_train


def Average(lst):
    return sum(lst) / len(lst)


def test():
    model = nb.NaiveBayesModel()
    path = 'E:/DATA/Sem8/fyp/Training.csv'
    final_df, df = model.extract('E:/DATA/Sem8/fyp/merge.csv')
    count = 0
    start = -200
    end = 0
    accuracy = []
    precision = []
    recall = []
    fscore = []
    stats = em.Evaluate()
    for count in range(5):
        df_test, df_train = split(final_df, start+200, end+200)
        print(df_train)
        li_clean_text = model.clean_data(df_train)
        uniqueWords = model.make_unique_li(li_clean_text)
    # # print(uniqueWords)
        docVector = model.binary_docvector(final_df, uniqueWords)
        df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
        predict_df, test_data = model.Predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI)
        # print("--------------Naive Bayes Accuracy Stats---------------------------")
        TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
        accuracy.append(stats.Accuracy(TP, TN, FP, FN))
        precision.append(stats.Precision(TP, FP))
        recall.append(stats.Recall(TP, FN))
        fscore.append(stats.fScore(TP, FN, FP))
        # print("---------------------------------------------------------------------")
    print("accuracy = ",Average(accuracy))
    print("precison = ", Average(precision))
    print("recall = ", Average(recall))
    print("f-score = ", Average(fscore))

# text_blob()
# naive_bayes()
# binary_naive_bayes()
test()