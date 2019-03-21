import pandas as pd
import numpy as np
import nltk

fd= open('E:/DATA/Sem8/fyp/AnnotatedData2.csv',encoding="utf-8", errors='replace')
df = pd.read_csv(fd)
print(df)

# create series of true and false. True is assigned for yes/no. False for undefine
defined=df['class'] != ("undefined")
# #output dataframe without undeined
df2=df[defined]
defined1=df2['class'] !="Undefined"
df4=df2[defined1]
#replace no PI with no
df3=df4.replace("No PI","no")
#replace PI with yes
final=df3.replace("PI","yes")

replace_yes=final.replace("Yes","yes")
final_df= replace_yes.replace("No","no")

print(final_df)

# create series of true and false. False is assigned for yes/no. True is for undefined
undefined=df['class'] == "undefined" 
#output dataframe without undeined
df_undefine=df[undefined]
print(df_undefine)

def text_concat():
    text=""
    for x in final_df["text"]:
        text=text +str(x)
    return text
corpus=text_concat()

import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

def removeStopWords(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w.lower()) 
    #return list of corpus without stop words in a list.        
    return filtered_sentence  

def removePunc(eachText):
    remove_punc = re.sub(r'[^\w\s]','',eachText)
    #return corpus with out punctuation.
    return remove_punc



def clean_data(corpus):
    remove_punc=removePunc(corpus)
    li_remove_stopWords= removeStopWords(remove_punc)
    return li_remove_stopWords

li_cleaned_text = clean_data(corpus)
# bprint(li_cleaned_text)

def make_unique_li(li_cleanText):
    unique_words_set= set(li_cleanText)
    unique_word_li = list(unique_words_set)
    return unique_word_li


def stemmed(li_cleanText):
    count_stemed=0
    for word in li_cleanText:
        if word[-1]=="s":
            li_cleanText[count_stemed]=word[:-1]
        elif word[-2:]=="ed":
            li_cleanText[count_stemed]=word[:-2]
        elif word[-3:]=="ing":
            li_cleanText[count_stemed]=word[:-3]
        count_stemed+=1  
    return li_cleanText   
# li_stemmed = stemmed(li_cleaned_text)
uniqueWords = make_unique_li(li_cleaned_text)
print(uniqueWords)

def DocVector():
    data=np.zeros([final_df['class'].count(), len(uniqueWords)])
    docVector1=pd.DataFrame(data,columns=uniqueWords)
    docVector=docVector1.assign(PurchaseIntention = list(final_df['class']))
    # docVector['Purchase Intention'] = final_df['class']
    # print(docVector['PurchaseIntention'])
    doc_count = 0
    for doc in final_df['text']:
        words = doc.split()
        for word in words:
            temp=word.lower()
            if temp in docVector.columns:
                docVector.at[doc_count, temp] += 1
        doc_count += 1
        
    return docVector


docVector = DocVector()

def WordGivenNoPI(tempNegDocVector):
    data=np.zeros([1,len(uniqueWords)])
    wordGivenNoPI=pd.DataFrame(data,columns=uniqueWords)
    columnSum = tempNegDocVector.sum(axis = 1, skipna = True)
    numWordsInNoPI = columnSum.sum()
    
    for word in uniqueWords:
        nk_wordinNoPI = tempNegDocVector[word].sum()
        wordGivenNoPI.at[0,word] = (nk_wordinNoPI+1) / (numWordsInNoPI+len(uniqueWords))
    return  (wordGivenNoPI , numWordsInNoPI)
    
    
def TrainModel():
    yesCount=docVector["PurchaseIntention"]=="yes"
    tempPosDocVector = docVector[yesCount]
    totalPI = tempPosDocVector["PurchaseIntention"].count()
    print("total PI ",totalPI)
    
    noCount=docVector["PurchaseIntention"]=="no"
    tempNegDocVector = docVector[noCount]
    # print(tempNegDocVector["PurchaseIntention"])
    totalNonPI = tempNegDocVector["PurchaseIntention"].count()
    print("total non PI ",totalNonPI)
    # print(totalPI+totalNonPI)
    # totalNonPI = docVector["PurchaseIntention"].count() - totalPI
    total = totalPI+totalNonPI
    Prob_PI = totalPI/total
    Prob_NoPI = totalNonPI/total
    
    data=np.zeros([1,len(uniqueWords)])
    wordGivenPI=pd.DataFrame(data,columns=uniqueWords)
    columnSum = tempPosDocVector.sum(axis = 1, skipna = True)
    numWordsInPI = columnSum.sum()
    
    for word in uniqueWords:
        nk_wordinPI = tempPosDocVector[word].sum()
        wordGivenPI.at[0,word] = (nk_wordinPI+1) / (numWordsInPI+len(uniqueWords))
        
    df_wordGivenNoPI , numWordsInNoPI= WordGivenNoPI(tempNegDocVector) 
    return (wordGivenPI,df_wordGivenNoPI,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI)
   

df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = TrainModel()
print("prob pos ",Prob_PI)
print("prob_neg ",Prob_NoPI)

def TestModel():
    tweet="I like buying iPhone x if they improve their camera"
    weighPI=Prob_PI
    weighNoPI=Prob_NoPI
    for word in tweet.lower().split():
        if word in uniqueWords:
            weighPI = weighPI * df_WordGivenPI.at[0,word]
            weighNoPI = weighNoPI * df_WordGivenNoPi.at[0,word]
        else:
             weighPI = weighPI * (1/(numWordsInPI+len(uniqueWords)))
             weighNoPI = weighNoPI * (1/(numWordsInNoPI+len(uniqueWords)))
    print("PI weight {0} No PI weight{1}".format(weighPI,weighNoPI))        
    if weighPI>weighNoPI:
        print("have PI")
    else:
        print("No PI")

    
TestModel()