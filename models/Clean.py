import pandas as pd
# import numpy as np
# import nltk
import re
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import math
import enchant
from stemming.porter2 import stem
import pathConfig as pc

#--------------------------------
###Path 
pathStopwords = pc.PATH_CONFIG['pathStopWords']
pathData = pc.PATH_CONFIG['pathData']
#-------------------------------


class DataCLean:

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
        stop_words = self.read_stopwords(pathStopwords)
        word_tokens = word_tokenize(text)
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w.lower())
                # return list of corpus without stop words in a list.
        return filtered_sentence

    def remove_stopwords(self, df_punc_remove):
        # stop_words = set(stopwords.words('english'))
        li_stopwords = self.read_stopwords(pathStopwords)
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
                            or lower_word == "don't" or lower_word == "hate":
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

    def check_english(self, temp_df):
        d = enchant.Dict("en_US")
        new_eng = pd.DataFrame()
        count = 0

        for sentence in temp_df['text']:
            temp_sent = ""
            for word in sentence.split():
                temp = word.lower()
                if d.check(word):
                    # print("ammar")
                    temp_sent = temp_sent + temp + " "
            # print(temp_sent)
            new_eng.at[count,'text'] = temp_sent
            new_eng.at[count, 'class'] = temp_df.iloc[count]['class']
            count += 1
        # print(new_eng)
        return new_eng

    def Stemming(self, temp_df):
        new_eng = pd.DataFrame()
        count = 0

        for sentence in temp_df['text']:
            temp_sent = ""
            for word in sentence.split():
                temp = stem(word.lower())
                temp_sent = temp_sent + temp + " "
            # print(temp_sent)
            new_eng.at[count, 'text'] = temp_sent
            new_eng.at[count, 'class'] = temp_df.iloc[count]['class']
            count += 1
        # print(new_eng)
        return new_eng

        # stem("factionally")

    def clean_data(self, final_df):
        # print(final_df)
        eng_df = self.check_english(final_df)
        # print(eng_df)
        df_stem = self.Stemming(eng_df)
        new_df = self.space(eng_df)
        new_corpus_df = self.handle_negation(new_df)
        remove_punc_df = self.remove_punc(new_corpus_df)
        df_remove_stopWords = self.remove_stopwords(remove_punc_df)

        new_corpus = self.text_concat(df_remove_stopWords)
        li_new_corpus = new_corpus.split()
        # print(remove_punc_df)
        return li_new_corpus, df_remove_stopWords

    def make_unique_li(self, li_cleanText):
        unique_words_set = set(li_cleanText)
        unique_word_li = list(unique_words_set)
        return unique_word_li

    def stemmed(self,li_cleanText):
        count_stemed = 0
        for word in li_cleanText:
            if word[-1] == "s":
                li_cleanText[count_stemed] = word[:-1]
            elif word[-2:] == "ed":
                li_cleanText[count_stemed] = word[:-2]
            elif word[-3:] == "ing":
                li_cleanText[count_stemed] = word[:-3]
            count_stemed += 1
        return li_cleanText

    def Clean(self):
        final_df, df = self.extract(pathData)
        # corpus = clean.text_concat(final_df)
        li_clean_text, df_clean = self.clean_data(final_df)
        # print("ammar")
        uniqueWords = self.make_unique_li(li_clean_text)
        return df_clean, uniqueWords
