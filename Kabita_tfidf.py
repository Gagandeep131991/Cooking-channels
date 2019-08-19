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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

data=pd.read_csv("kabitakitchen.csv",encoding = "ISO-8859-1")

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

#Removing spaces
data['commentText'] = data['commentText'].str.strip()
print(data['commentText'].head())

#Removing the punctuations, special character
data['commentText'] = data['commentText'].str.replace('[^a-z# _]','')
print(data['commentText'].head())

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

#make new file after preprocessing

data.to_csv('nishaPreprocessing.csv', index=False)

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

#tf-idf vectorizer

sentences = []

for text in data.commentText:
	sentences.append(text)
print(sentences)

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(sentences).toarray()
print(X)

#Splitting data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, data['Labels'], test_size=0.3, random_state=5)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)



#-----------------------Grid search for random forest---------------------


rfc=RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [10,20,30,40,50,60,70,80,90,100]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#----------Random forest classifier--------------------

classifier = RandomForestClassifier(n_estimators=100, random_state=0)

print((cross_val_score(classifier,X_train, y_train, cv=10)))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Random forest")
print(accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))


#-------------------Grid search for multinomial naive bayes--------

clf = MultinomialNB()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1,10,100,1000]
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#------------Multinomial Naive Bayes------------------

clf = MultinomialNB(alpha=1)
print((cross_val_score(clf,X_train, y_train, cv=10)))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Multinomial naive bayes")
print(accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))

#------------------Grid search for bernoulli naive bayes----------

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1,10,100,1000]
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#------------------Bernoulli Naive Bayes-----------------------



bnb = BernoulliNB(alpha=0.1)
print((cross_val_score(bnb,X_train, y_train, cv=10)))
bnb.fit(X_train, y_train)
y_pred=bnb.predict(X_test)
print("Bernoulli nayive bayes")
print(accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))



#-----------------Gaussian Naive Bayes--------------------
from sklearn.naive_bayes import GaussianNB

gnb= GaussianNB()
print((cross_val_score(gnb,X_train, y_train, cv=10)))
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)
print("Gaussian naive bayes")
print(accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))

#---------------------Grid search for logistic regression--------------------------------
from sklearn.linear_model import LogisticRegression

C = [0.001, 0.01, 0.1, 1,10,100,1000]
param_grid = dict(C=C)

import time

lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

#--------------Logistic Regression-------------------------

model = LogisticRegression(penalty='l2',C=10, dual=False, multi_class='multinomial', solver='newton-cg')
print((cross_val_score(model,X_train, y_train, cv=10)))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression")
print(accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))


#------------------------Grid search for Linear SVM-------------------------

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

lsvm=LinearSVC()
param_grid = {
    'C': [0.001, 0.01, 0.1, 1,10,100,1000]
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
clfr.fit(X_train, y_train)
y_pred = clfr.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("SVM Linear:",metrics.accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))


#------------------Grid search for polynomial svm

from sklearn.model_selection import GridSearchCV

pm=svm.SVC(kernel='poly')
param_grid = {
    'C': [0.001, 0.01, 0.1, 1,10,100,1000],
        'gamma':[0.001, 0.01, 0.1, 1,10,100,1000]
}
CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)

#----------------SVM(polynomial)--------------
from sklearn import svm
clfpol = svm.SVC(C=0.01, decision_function_shape='ovr', degree=3, gamma= 10, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print((cross_val_score(clfpol,X_train, y_train, cv=10)))

#Train the model using the training sets
clfpol.fit(X_train, y_train)

# #Predict the response for test dataset
y_pred = clfpol.predict(X_test)
print("SVM Polynomial:",metrics.accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))

#------------------------Grid search SVM(gaussian)--------------

pm=svm.SVC(kernel='rbf')
param_grid = {
    'C': [0.001, 0.01, 0.1, 1,10,100,1000],
        'gamma':[0.001, 0.01, 0.1, 1,10,100,1000]
}
CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train, y_train)

r=CV_rfc.best_params_
print(r)


#--------------SVM(Gaussian)---------------------------
clfgs = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print((cross_val_score(clfgs,X_train, y_train, cv=10)))

#Train the model using the training sets
clfgs.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clfgs.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("SVM gaussian:",metrics.accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))


#-----------------Decision tress-------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier(random_state=0)
print((cross_val_score(clf,X_train, y_train, cv=10)))
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("mcc", matthews_corrcoef(y_test, y_pred))
print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))
print ('Recall:', recall_score(y_test, y_pred,average='weighted'))
print ('Precision:', precision_score(y_test, y_pred, average='weighted'))




