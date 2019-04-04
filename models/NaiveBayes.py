import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math


class NaiveBayesModel:

    def extract(self, path):
        fd = open(path, encoding="utf-8", errors='replace')
        df = pd.read_csv(fd)
        defined = df['class'] != ("undefined")
        # #output dataframe without undeined
        df2 = df[defined]
        defined1 = df2['class'] != "Undefined"
        df4 = df2[defined1]
        # replace no PI with no
        df3 = df4.replace("No PI", "no")
        # replace PI with yes
        final = df3.replace("PI", "yes")

        replace_yes = final.replace("Yes", "yes")
        final_df = replace_yes.replace("No", "no")
        return final_df, df

    def undefined_extract(self, df):
        undefined = df['class'] == "undefined"
        # output dataframe without undeined
        df_undefine = df[undefined]
        print(df_undefine)
        return df_undefine

    def text_concat(self, final_df):
        text = ""
        for x in final_df["text"]:
            text = text + str(x)
        return text

    def read_stopwords(self, path):
        file1 = open(path, "r")
        stopword = file1.readlines()
        file1.close()
        li_stopwords = stopword[0].split()
        return li_stopwords


    def removeStopWords(self,text):
        # stop_words = set(stopwords.words('english'))
        stop_words = self.read_stopwords("E:/DATA/Sem8/fyp/stopwords.txt")
        word_tokens = word_tokenize(text)
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w.lower())
                # return list of corpus without stop words in a list.
        return filtered_sentence

    def remove_stopwords(self, df_punc_remove):
        # stop_words = set(stopwords.words('english'))
        li_stopwords = self.read_stopwords("E:/DATA/Sem8/fyp/stopwords.txt")
        # print(stop_words)
        count_clean = 0
        for text in df_punc_remove['text']:
            word_tokens = word_tokenize(text)
            clean_text = ""
            for w in word_tokens:
                if w.lower() not in li_stopwords:
                    clean_text = clean_text + w.lower() + ' '
            df_punc_remove.at[count_clean, 'text'] = clean_text
            df_punc_remove.at[count_clean, 'class'] = df_punc_remove.iloc[count_clean]['class']
            count_clean += 1
        # return list of corpus without stop words in a list.
        # print(df_punc_remove)
        return df_punc_remove

    def removePunc(self, eachText):
        remove_punc = re.sub(r'[^\w\s]', '', eachText)
        return remove_punc
        # pattern = re.compile(r'[a-zA-Z]+')
        # matches = pattern.finditer(eachText)
        # new_corpus = ""
        # for match in matches:
        #     new_corpus = new_corpus + match.group() + " "
        # return new_corpus

    def remove_punc(self, temp_df):
        count = 0
        for text in temp_df['text']:
            out = re.sub(r'[^\w\s]', '', text)
            temp_df.at[count,'text'] = out
            temp_df.at[count, 'class'] = temp_df.iloc[count]['class']
            count += 1
        return temp_df

    def space(self, final_df):
        new_df = pd.DataFrame()
        count_tweets = 0
        for text in final_df['text']:
            temp = ""
            for char in text:
                if char in [",",".","!","?",":",";"]:
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

    def handle_negation(self, final_df):
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
                        for i in range(temp,len(li_text)):
                            if li_text[i] in [",","?","!","."]:
                                temp_text = " "+temp_text + li_text[i] + " "
                                break
                            else:
                                temp_text = temp_text + "NOT_" + li_text[i]+" "

                    else:
                        temp_text = temp_text + word + " "
                # print(temp_text)
                out_df.at[count_tweet, 'text'] = temp_text
                out_df.at[count_tweet,'class'] = final_df.iloc[count_tweet]['class']
                count_tweet += 1
            return out_df

    def clean_data(self, final_df):
        # print(final_df)
        new_df = self.space(final_df)
        # print("Hello clean data")
        # print(new_df["text"])
        new_corpus_df = self.handle_negation(new_df)
        # print(new_corpus_df['text'])
        new_corpus = self.text_concat(new_corpus_df)

        remove_punc = self.removePunc(new_corpus)
        li_remove_stopWords = self.removeStopWords(remove_punc)
        # print(li_remove_stopWords)
        return li_remove_stopWords

    def make_unique_li(self, li_cleanText):
        unique_words_set = set(li_cleanText)
        unique_word_li = list(unique_words_set)
        return unique_word_li

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
        print(docVector['good'])
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


    def Predict(self, Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI):
        test_path = "E:/DATA/Sem8/fyp/test data/Testing.csv"
        test_data, test_df = self.extract(test_path)
        new_df = self.space(test_data)
        new_corpus_df = self.handle_negation(new_df)
        punc_df = self.remove_punc(new_corpus_df)
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

