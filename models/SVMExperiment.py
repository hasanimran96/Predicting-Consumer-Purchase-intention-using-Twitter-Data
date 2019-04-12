import pandas as pd
import Clean as cl
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import model_selection, naive_bayes
import numpy as np

def SVM(path):
    _dcl = cl.DataCLean()
    final_df, df = _dcl.extract(path)
    # corpus = model.text_concat(final_df)
    li_clean_text = _dcl.clean_data(final_df)
    uniqueWords = _dcl.make_unique_li(li_clean_text)
    # print(uniqueWords)
    docVector = _dcl.DocVector(final_df, uniqueWords) ###_dcl.DocVector or _dcl.binary_docvectir
    ###########################
    df = docVector.values
    X_train,Y = df[:,:-1],df[:,-1]
    Y_train = convert_to_0_or_1(Y);
    return (X_train, Y_train)
"""
###Merge file function
def MergeFile():
    filenames = ['Training.csv', 'Testing.csv']
    with open('MergeFile.csv', 'w',encoding='latin') as outfile:
        for fname in filenames:
            with open(fname,encoding='latin') as infile:
                outfile.write(infile.read())

#MergeFile()
"""

def convert_to_0_or_1(Y):
    Y_train = []
    for y in Y:
        if y == 'yes':
            Y_train.append(1)
        else:
            Y_train.append(0)
    return Y_train



##Training

path = '../data/AnnotatedData3.csv'
X, Y = SVM(path)

a = np.size(X,0)
X_split = int(np.size(X,0)*0.7)


X_train = X[0:X_split]
Y_train = Y[0:X_split]

X_test = X[X_split:,: ]
Y_test = Y[X_split : ]



#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

class_weight = {0: 2,
                1: 1,}

#print(Y_train)
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
model = clf.fit(X_train, Y_train)
print(model.score(X_test,Y_test))

Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train, Y_train)
print(Naive.score(X_test,Y_test))

