import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from sklearn import metrices
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data=pd.read_csv("Nishafin.csv")

shape_data=data.shape
print(shape_data)

#Finding the null values
null_data=data.isnull()#Check all null value in dataset
print(null_data)
sumnull_data=data.isnull().sum() #check how many null values present in each column
print(sumnull_data)

#Count number of stopwords in each comment
stop = stopwords.words('english')
data['stopwords'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x in stop]))
print(data[['commentText','stopwords']].head())

#Removing the stopwords in each comment
data['commentText'] = data['commentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data['commentText'].head())

#Identifying the number of upper case
data['upper'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
print(data[['commentText','upper']].head())

#Converting in lower case
data['commentText'] = data['commentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(data['commentText'].head())

#count number of special character
data['hastags'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
print(data[['commentText','hastags']].head())

#Removing the punctuations, special characters
data['commentText'] = data['commentText'].str.replace('[^a-z# _]','')
print(data['commentText'].head())

#---------------------------------EDA-------------------------------------------
#Count the number of words in each comment
data['word_count'] = data['commentText'].apply(lambda x: len(str(x).split(" ")))
print(data['word_count'])
print(data[['commentText','word_count']].head())

#Count the number of characters
data['char_count'] = data['commentText'].str.len() #It includes spaces too
print(data[['commentText','char_count']].head())

#Average words
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['commentText'].apply(lambda x: avg_word(x))
print(data[['commentText','avg_word']].head())

#count the numberic values in each comment
data['numerics'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(data[['commentText','numerics']].head())

data.to_csv('nis.csv', index=False)

#Tokenization
tokens=data['commentText'].apply(lambda x: x.split())
print(tokens)

#stemming
stemmer=PorterStemmer()
tokens_stem=tokens.apply(lambda x: [stemmer.stem(i) for i in x])
print(tokens_stem)

#Lemmatization

data['commentText'] = data['commentText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(data['commentText'].head())

#Wordcloud

import numpy as np # linear algebra
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
#
mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10
mpl.rcParams['savefig.dpi']=100             #72
mpl.rcParams['figure.subplot.bottom']=.1

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40,
                          random_state=42
                         ).generate(str(data['commentText']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

#tf-idf vectorization

sentences = []

for text in data.commentText:
	sentences.append(text)
print(sentences)

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(sentences).toarray()
print(X)

# Dividing into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, data['Labels'], test_size=0.3, random_state=5)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)



#-----------------------Grid search for random forest---------------------

from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [10,20,30,40,50,60,70,80,90,100]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)


#----------Random forest classifier--------------------
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

classifier = RandomForestClassifier(n_estimators=80, random_state=0)
trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train)
print(accuracy_score(y_train,trained_model.predict(X_train)))
print(recall_score(y_train,trained_model.predict(X_train),average='weighted'))

print((cross_val_score(classifier,X_train,y_train,cv=10)))
# 70.52
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))#70.61
print("mcc", matthews_corrcoef(y_test, y_pred))
#65.89
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#70.61
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#70.61
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#71.63
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Random forest")
print(accuracy_score(y_test, y_pred))
#70.61


#-------------------Grid search for multinomial naive bayes--------

from sklearn.model_selection import GridSearchCV

clf = MultinomialNB()
param_grid = {
    'alpha': [0.1,0.001,0.0001,0.01,0.00001,1,10,100,1000,10000]
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#------------Multinomial Naive Bayes------------------

clf = MultinomialNB(alpha=1)
print((cross_val_score(clf,X_train, y_train, cv=10)))
#68.97
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Multinomial naive bayes")
print(accuracy_score(y_test, y_pred))
#70.27
print("mcc", matthews_corrcoef(y_test, y_pred))
#65.64
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#69.57
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#70.27
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#71.13

#------------------Grid search for bernoulli naive bayes----------

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
#
clf = BernoulliNB()
param_grid = {
    'alpha': [0.1,0.001,0.0001,0.01,0.00001,1,10,100,1000,10000]
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#------------------Bernoulli Naive Bayes-----------------------



bnb = BernoulliNB(alpha=0.1)
print((cross_val_score(bnb,X_train, y_train, cv=10)))
#67.52
bnb.fit(X_train, y_train)
y_pred=bnb.predict(X_test)
print("Bernoulli nayive bayes")
print(accuracy_score(y_test, y_pred))
#67.75
print("mcc", matthews_corrcoef(y_test, y_pred))
#62.63
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#66.22
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#67.75
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#66.50


#-----------------Gaussian Naive Bayes--------------------
from sklearn.naive_bayes import GaussianNB

gnb= GaussianNB()
print((cross_val_score(gnb,X_train, y_train, cv=10)))#53.61(without)
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)
print("Gaussian naive bayes")
print(accuracy_score(y_test, y_pred))
# #54.96(without)

print("mcc", matthews_corrcoef(y_test, y_pred))
#47.27
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#52.83
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#54.69
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#52.10

#---------------------Grid search for logistic regression--------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#dual=[True,False]
#max_iter=[100,110,120,130,140,150]
C = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
param_grid = dict(C=C)

import time

lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

param_grid = {
    'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
    }
CV_rfc = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#--------------Logistic Regression-------------------------
model = LogisticRegression(penalty='l2',C=10, dual=False, multi_class='multinomial', solver='newton-cg')
print((cross_val_score(model,X_train, y_train, cv=10)))#73.03
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression")
print(accuracy_score(y_test, y_pred))

#73.46
print("mcc", matthews_corrcoef(y_test, y_pred))
#69.15
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#73.73
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#73.46
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#74.54

#------------------------Grid search for Linear SVM-------------------------

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

lsvm=LinearSVC()
param_grid = {
    'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
    }
CV_rfc = GridSearchCV(estimator=lsvm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)


#-------------------Linear Support Vector Machine------------------

from sklearn import svm

clfr = svm.SVC(C=1, decision_function_shape='ovr', kernel='linear',
      probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
print((cross_val_score(clfr,X_train, y_train, cv=10)))
#73.47
clfr.fit(X_train, y_train)
y_pred = clfr.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("SVM Linear:",metrics.accuracy_score(y_test, y_pred))
#73.74

print("mcc", matthews_corrcoef(y_test, y_pred))
#69.48
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#73.94
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#73.67
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#75.15


#------------------Grid search for polynomial svm
#
from sklearn.model_selection import GridSearchCV
#
pm=svm.SVC(kernel='poly')
param_grid = {
    'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0],
        'gamma':[1.0,10.0,100.0,1000.0,10000.0,100000.0]
}
CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#----------------SVM(polynomial)--------------
from sklearn import svm
clfpol = svm.SVC(C=0.1, decision_function_shape='ovr', degree=3, gamma= 100, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print((cross_val_score(clfpol,X_train, y_train, cv=10)))
#60.99
#Train the model using the training sets
clfpol.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clfpol.predict(X_test)
print("SVM Polynomial:",metrics.accuracy_score(y_test, y_pred))
#60.95
print("mcc", matthews_corrcoef(y_test, y_pred))
#56.69
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#62.84
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#60.95
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#72.33


#------------------------Grid search SVM(gaussian)--------------
from sklearn.model_selection import GridSearchCV

pm=svm.SVC(kernel='rbf')
param_grid = {
    'C': [1, 10,100,1000],
        'gamma':[0.001, 0.01, 0.1, 1]
}
CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)




# #--------------SVM(Gaussian)---------------------------
clfgs = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print((cross_val_score(clfgs,X_train, y_train, cv=10)))
#0.7239
#Train the model using the training sets
clfgs.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clfgs.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("SVM gaussian:",metrics.accuracy_score(y_test, y_pred))
#73.33

print("mcc", matthews_corrcoef(y_test, y_pred))
#68.98
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#73.42
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#73.33
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#74.03

#-----------------Decision tress-------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=1)
print((cross_val_score(clf,X_train, y_train, cv=10)))
#64.61(without)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#64.89(without)

print("mcc", matthews_corrcoef(y_test, y_pred))
# #59.12
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#64.88
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#64.89
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#65.28

#--------------------- grid seach for decision tree
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier()
param_grid = [{'max_depth':np.arange(1, 21),
              'min_samples_leaf':[1, 10, 20, 30, 40, 50,60,70,80,90,100]}]

CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)
