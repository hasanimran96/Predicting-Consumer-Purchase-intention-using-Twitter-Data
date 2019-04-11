import pandas as pd
import numpy as np
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import math


class NaiveBayesModel:

    def undefined_extract(self, df):
        undefined = df['class'] == "undefined"
        # output dataframe without undeined
        df_undefine = df[undefined]
        print(df_undefine)
        return df_undefine


    def DocVector(self, final_df, uniqueWords):
        data = np.zeros([final_df['class'].count(), len(uniqueWords)])
        docVector1 = pd.DataFrame(data, columns=uniqueWords)
        docVector = docVector1.assign(PurchaseIntention=list(final_df['class']))
        # docVector['Purchase Intention'] = final_df['class']
        # print(docVector['PurchaseIntention'])
        doc_count = 0
        for doc in final_df['text']:
            words = doc.split()
            for word in words:
                temp = word.lower()
                if temp in docVector.columns:
                    docVector.at[doc_count, temp] += 1
            doc_count += 1

        return docVector

    def binary_docvector(self, final_df, uniqueWords):
        data = np.zeros([final_df['class'].count(), len(uniqueWords)])
        docVector1 = pd.DataFrame(data, columns=uniqueWords)
        docVector = docVector1.assign(PurchaseIntention=list(final_df['class']))
        # docVector['Purchase Intention'] = final_df['class']
        # print(docVector['PurchaseIntention'])
        doc_count = 0
        for doc in final_df['text']:
            words = doc.split()
            for word in words:
                temp = word.lower()
                if temp in docVector.columns:
                    if docVector.iloc[doc_count][temp] < 1:
                        docVector.at[doc_count, temp] += 1
            doc_count += 1
        # print(docVector['good'])
        return docVector

    def WordGivenNoPI(self, tempNegDocVector, uniqueWords):
        data = np.zeros([1, len(uniqueWords)])
        wordGivenNoPI = pd.DataFrame(data, columns=uniqueWords)
        columnSum = tempNegDocVector.sum(axis=1, skipna=True)
        numWordsInNoPI = columnSum.sum()

        for word in uniqueWords:
            nk_wordinNoPI = tempNegDocVector[word].sum()
            wordGivenNoPI.at[0, word] = (nk_wordinNoPI + 1) / (numWordsInNoPI + len(uniqueWords))
        return (wordGivenNoPI, numWordsInNoPI)

    def TrainModel(self, Train_Vector, uniqueWords):
        yesCount = Train_Vector["PurchaseIntention"] == "yes"
        tempPosDocVector = Train_Vector[yesCount]
        totalPI = tempPosDocVector["PurchaseIntention"].count()
        print("total PI ", totalPI)

        noCount = Train_Vector["PurchaseIntention"] == "no"
        tempNegDocVector = Train_Vector[noCount]
        # print(tempNegDocVector["PurchaseIntention"])
        totalNonPI = tempNegDocVector["PurchaseIntention"].count()
        print("total non PI ", totalNonPI)
        # print(totalPI+totalNonPI)
        # totalNonPI = docVector["PurchaseIntention"].count() - totalPI
        total = totalPI + totalNonPI
        Prob_PI = totalPI / total
        Prob_NoPI = totalNonPI / total

        data = np.zeros([1, len(uniqueWords)])
        wordGivenPI = pd.DataFrame(data, columns=uniqueWords)
        columnSum = tempPosDocVector.sum(axis=1, skipna=True)
        numWordsInPI = columnSum.sum()

        for word in uniqueWords:
            nk_wordinPI = tempPosDocVector[word].sum()
            wordGivenPI.at[0, word] = (nk_wordinPI + 1) / (numWordsInPI + len(uniqueWords))

        df_wordGivenNoPI, numWordsInNoPI = self.WordGivenNoPI(tempNegDocVector, uniqueWords)
        return wordGivenPI, df_wordGivenNoPI, Prob_PI, Prob_NoPI, numWordsInPI, numWordsInNoPI


    def Predict_kfold(self, Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,df_test, clean):
        new_df = clean.space(df_test)
        new_corpus_df = clean.handle_negation(new_df)
        punc_df = clean.remove_punc(new_corpus_df)
        # print(punc_df)
        # test_data = test_data.assign(PredictedClass= list(test_data['text']))
        # test_data = test_data[['class', 'PredictedClass', 'text']]
        # print(test_data["text"].count())
        predict_df = pd.DataFrame()
        weighPI = Prob_PI
        weighNoPI = Prob_NoPI
        count_test = 0
        for sentence in punc_df['text']:
            # print(count_test)
            for word in sentence.lower().split():
                if word in uniqueWords:
                    weighPI = weighPI * df_WordGivenPI.at[0, word]
                    weighNoPI = weighNoPI * df_WordGivenNoPi.at[0, word]
                else:
                    weighPI = weighPI * (1 / (numWordsInPI + len(uniqueWords)))
                    weighNoPI = weighNoPI * (1 / (numWordsInNoPI + len(uniqueWords)))
            predict_df.at[count_test, 'WeightPI'] = weighPI
            if weighPI > weighNoPI:
                predict_df.at[count_test, 'PredictedClass'] = 'yes'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])
            else:
                predict_df.at[count_test, 'PredictedClass'] = 'no'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])

            count_test += 1
            weighPI = Prob_PI
            weighNoPI = Prob_NoPI
        return predict_df, punc_df

    def Predict(self, Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,clean):
        test_path = "E:/DATA/Sem8/fyp/test data/Testing.csv"
        test_data, test_df = clean.extract(test_path)
        new_df = clean.space(test_data)
        new_corpus_df = clean.handle_negation(new_df)
        punc_df = clean.remove_punc(new_corpus_df)
        # print(punc_df)

        # test_data = test_data.assign(PredictedClass= list(test_data['text']))
        # test_data = test_data[['class', 'PredictedClass', 'text']]
        # print(test_data["text"].count())
        predict_df = pd.DataFrame()
        weighPI = Prob_PI
        weighNoPI = Prob_NoPI
        count_test = 0
        for sentence in punc_df['text']:
            # print(count_test)
            for word in sentence.lower().split():
                if word in uniqueWords:
                    weighPI = weighPI * df_WordGivenPI.at[0, word]
                    weighNoPI = weighNoPI * df_WordGivenNoPi.at[0, word]
                else:
                    weighPI = weighPI * (1 / (numWordsInPI + len(uniqueWords)))
                    weighNoPI = weighNoPI * (1 / (numWordsInNoPI + len(uniqueWords)))
            predict_df.at[count_test, 'WeightPI'] = weighPI
            if weighPI > weighNoPI:
                predict_df.at[count_test, 'PredictedClass'] = 'yes'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])
            else:
                predict_df.at[count_test, 'PredictedClass'] = 'no'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])

            count_test += 1
            weighPI = Prob_PI
            weighNoPI = Prob_NoPI
        return predict_df, punc_df

    def tf_idf(self, corpus, df_cleaned_text):
        unique = list(set(corpus.split()))
        # print(unique)
        # tfIdf_df = pd.DataFrame(columns=unique)
        tf_df = self.DocVector(df_cleaned_text,unique)
        # idf_df = pd.DataFrame()
        total_docs = len(tf_df.index)
        for column in unique:
            num_doc_word = 0
            for no_doc in range(total_docs):
                if tf_df.at[no_doc, column] != 0:
                    num_doc_word += 1
            idf = math.log(total_docs / num_doc_word)
            # idf_df.at[0,column] = idf
            tf_df[column] = tf_df[column].multiply(idf)
            idf = 0
        # print(tf_df['buy'])
        return tf_df

