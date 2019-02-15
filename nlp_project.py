import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

origdata = pd.read_csv('C:/Users/hasan/Desktop/Reviews.csv',
                       usecols=['Score', 'Summary', 'Text'], nrows=25000)
# print(origdata.head(3))


def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'


Score = origdata['Score']
Score = Score.map(partition)
Text = origdata['Text']
Summary = origdata['Summary']
X_train, X_test, y_train, y_test = train_test_split(
    Summary, Score, test_size=0.2, random_state=42)

intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)
stemmer = PorterStemmer()


def cleandata(eachText):
    eachText = eachText.lower()
    eachText = eachText.translate(trantab)
    tokens = nltk.word_tokenize(eachText)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmed = []
    for eachitem in tokens:
        stemmed.append(stemmer.stem(eachitem))
    eachText = ' '.join(stemmed)
    return eachText


corpus = []
for eachi in X_train:
    corpus.append(cleandata(eachi))

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

test_set = []
for eachi in X_test:
    test_set.append(cleandata(eachi))

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

df = pd.DataFrame({'Before': X_train, 'After': corpus})
print(df.head(20))

prediction = dict()

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg_result = logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)


def formatt(x):
    if x == 'negative':
        return 0
    return 1


vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate,
             colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

words = count_vect.get_feature_names()
feature_coefs = pd.DataFrame(
    data=list(zip(words, logreg_result.coef_[0])),
    columns=['feature', 'coef'])

feature_coefs.sort_values(by='coef')
print(feature_coefs)
