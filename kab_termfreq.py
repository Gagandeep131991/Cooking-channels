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

import matplotlib.pyplot as plt

data=pd.read_csv("kabitakitchen.csv",encoding = "ISO-8859-1")

shape_data=data.shape
#print(shape_data)

#Finding the null values
null_data=data.isnull()#Check all null value in dataset
#print(null_data)
sumnull_data=data.isnull().sum() #check how many null values present in each column
#print(sumnull_data)

#newdata=data.fillna(" ") #Fill all null or empty cells in original DataFrame with an empty space and set that to a new DataFrame variabl
#newdata.isnull().sum()  #Verify that you no longer have any null values by running
#modifieddata.to_csv('nishafinal.csv',index=False) #Save modified dataset to a new CSV
#md=pd.read_csv("nishafinal.csv") #save modified csv into new variable in order to perform further operation

#Count number of stopwords in each comment
stop = stopwords.words('english')

data['stopwords'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x in stop]))
#print(data[['commentText','stopwords']].head())
# #
# #Removing the stopwords in each comment
data['commentText'] = data['commentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#print(data['commentText'].head())
#
#Identifying the number of upper case
data['upper'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
#print(data[['commentText','upper']].head())
#
#
#Converting in lower case
data['commentText'] = data['commentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#print(data['commentText'].head())

#count number of special character
data['hastags'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
#print(data[['commentText','hastags']].head())
#
#Removing spaces
data['commentText'] = data['commentText'].str.strip()
#print(data['commentText'].head())
#
# #Removing the punctuations, special character
data['commentText'] = data['commentText'].str.replace('[^a-z# _]','')
#print(data['commentText'].head())
#
#Count the number of words in each comment
data['word_count'] = data['commentText'].apply(lambda x: len(str(x).split(" ")))
# print(data['word_count'])
# print(data[['commentText','word_count']].head())
# #
#Count the number of characters
data['char_count'] = data['commentText'].str.len() #It includes spaces too
#print(data[['commentText','char_count']].head())
# #
#Average words
# def avg_word(sentence):
#   words = sentence.split()
#   return (sum(len(word) for word in words)/len(words))
#
# data['avg_word'] = data['commentText'].apply(lambda x: avg_word(x))
# print(data[['commentText','avg_word']].head())
# #
#
# #
#count the numberic values in each comment
data['numerics'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#print(data[['commentText','numerics']].head())
#
#
#
# #data.to_csv('nishaTestingfin.csv', index=False)
#
tokens=data['commentText'].apply(lambda x: x.split())
#print(tokens)

#stemming
stemmer=PorterStemmer()

tokens_stem=tokens.apply(lambda x: [stemmer.stem(i) for i in x])

#print(tokens_stem)
#
data['commentText'] = data['commentText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(data['commentText'].head())
#
#
sentences = []

for text in data.commentText:
	sentences.append(text)
#print(sentences)
#
vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
X = vectorizer.fit_transform(sentences).toarray()

#X_train = vectorizer.fit_transform(sentences).toarray()
#print(X)
#
# #X = X.todense()
#
X_train, X_test, y_train, y_test = train_test_split(X, data['Labels'], test_size=0.3, random_state=5)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)



#-----------------------Grid search for random forest---------------------
# #
# from sklearn.model_selection import GridSearchCV
#
# rfc=RandomForestClassifier(random_state=0)
# param_grid = {
#     'n_estimators': [10,20,30,40,50,60,70,80,90,100]
# }
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)
# n_estimators=100


#----------Random forest classifier--------------------
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# classifier = RandomForestClassifier(n_estimators=100, random_state=0)
#
# print((cross_val_score(classifier,X_train, y_train, cv=10)))



#70.92
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# #print(y_pred)
# #
# #
# #
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print("Random forest")
# print(accuracy_score(y_test, y_pred))
# #72.31
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98


#-------------------Grid search for multinomial naive bayes--------

# from sklearn.model_selection import GridSearchCV
# #
# clf = MultinomialNB()
# param_grid = {
#     'alpha': [0.1,0.001,0.0001,0.01,0.00001,1,10,100,1000,10000]
# }
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)
#alpha=0.1
#------------Multinomial Naive Bayes------------------
# #
# clf = MultinomialNB(alpha=0.1)
# print((cross_val_score(clf,X_train, y_train, cv=10)))
# # #71.97
# [0.72543353 0.75144509 0.74202899 0.70930233 0.72093023 0.72886297
#  0.71345029 0.76470588 0.70882353 0.70588235]
# [0.69653179 0.74855491 0.76521739 0.69476744 0.71802326 0.72303207
# Traceback (most recent call last):
#  0.70467836 0.72647059 0.70294118 0.71764706]
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Multinomial naive bayes")
# print(accuracy_score(y_test, y_pred))
# #71.29
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98

#------------------Grid search for bernoulli naive bayes----------

from sklearn.naive_bayes import BernoulliNB
# from sklearn.model_selection import GridSearchCV
# #
# clf = BernoulliNB()
# param_grid = {
#     'alpha': [0.1,0.001,0.0001,0.01,0.00001,1,10,100,1000,10000]
# }
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)
#0.1
#------------------Bernoulli Naive Bayes-----------------------



# bnb = BernoulliNB(alpha=0.1)
# print((cross_val_score(bnb,X_train, y_train, cv=10)))
# #68.30
# bnb.fit(X_train, y_train)
# y_pred=bnb.predict(X_test)
# print("Bernoulli nayive bayes")
# print(accuracy_score(y_test, y_pred))
# #66.87
#
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98

#-----------------Gaussian Naive Bayes--------------------
# from sklearn.naive_bayes import GaussianNB
#
# gnb= GaussianNB()
# print((cross_val_score(gnb,X_train, y_train, cv=10)))#51.58
# gnb.fit(X_train, y_train)
# y_pred=gnb.predict(X_test)
# print("Gaussian naive bayes")
# print(accuracy_score(y_test, y_pred))
# #0.5
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#
#---------------------Grid search for logistic regression--------------------------------
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# #dual=[True,False]
# #max_iter=[100,110,120,130,140,150]
# C = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
# param_grid = dict(C=C)
#
# import time
#
# lr = LogisticRegression(penalty='l2')
# grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)
#
# start_time = time.time()
# grid_result = grid.fit(X_train, y_train)
# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# print("Execution time: " + str((time.time() - start_time)) + ' ms')
#C=10
#--------------Logistic Regression-------------------------
#
# #
# #
# model = LogisticRegression(penalty='l2',C=10, dual=False, multi_class='multinomial', solver='newton-cg')
# print((cross_val_score(model,X_train, y_train, cv=10)))#75.94
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Logistic Regression")
# print(accuracy_score(y_test, y_pred))
# #75.37
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98

#------------------------Grid search for Linear SVM-------------------------

from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
#
# lsvm=LinearSVC()
# param_grid = {
#     'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
#     }
# CV_rfc = GridSearchCV(estimator=lsvm, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)
#1.0


#-------------------Support Vector Machine------------------

# from sklearn import svm
#
# clfr = svm.SVC(C=1, decision_function_shape='ovr', kernel='linear',
#       probability=False, random_state=None, shrinking=True,
#       tol=0.001, verbose=False)
# #
# print((cross_val_score(clfr,X_train, y_train, cv=10)))#76.40
# clfr.fit(X_train, y_train)
# y_pred = clfr.predict(X_test)
# # # # #
# from sklearn import metrics
# # # # #
# # # Model Accuracy: how often is the classifier correct?
# print("SVM Linear:",metrics.accuracy_score(y_test, y_pred))
# # 74.21
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#


#------------------Grid search for polynomial svm
#
# from sklearn.model_selection import GridSearchCV
# #
# pm=svm.SVC(kernel='poly')
# param_grid = {
#     'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0],
#         'gamma':[1.0,10.0,100.0,1000.0,10000.0,100000.0]
# }
# CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)

# #----------------SVM(polynomial)--------------
from sklearn import svm
# clfpol = svm.SVC(C=0.01, decision_function_shape='ovr', degree=3, gamma= 10, kernel='poly',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print((cross_val_score(clfpol,X_train, y_train, cv=10)))
# #65.07
# # #Train the model using the training sets
# clfpol.fit(X_train, y_train)
# #
# # #Predict the response for test dataset
# y_pred = clfpol.predict(X_test)
# print("SVM Polynomial:",metrics.accuracy_score(y_test, y_pred))
# # #64.14
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#

#c=0.001, gamma=10---65.14,64.89
#C=0.01,      10----64.96
#C=0.1,         ----64.96
#C=0.001,       1---19
#C=0.01        100--64.96
#------------------------Grid search SVM(gaussian)--------------
# from sklearn.model_selection import GridSearchCV
# # #
# pm=svm.SVC(kernel='rbf')
# param_grid = {
#     'C': [1, 10,100,1000],
#         'gamma':[0.001, 0.01, 0.1, 1]
# }
# CV_rfc = GridSearchCV(estimator=pm, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)




# # #--------------SVM(Gaussian)---------------------------
# clfgs = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print((cross_val_score(clfgs,X_train, y_train, cv=10)))
# [0.76300578 0.77456647 0.80289855 0.72383721 0.74127907 0.74635569
#  0.75438596 0.73823529 0.70882353 0.74411765]
# #76.32
# #Train the model using the training sets
# clfgs.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred = clfgs.predict(X_test)
#
# from sklearn import metrics
#
# # Model Accuracy: how often is the classifier correct?
# print("SVM gaussian:",metrics.accuracy_score(y_test, y_pred))
# #75.30
#
#
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98

#C=1000,gamma=0.001-----74.01
#  1000,      0.01-------73.80
#  100,       0.001-----69.31
#  1000,        0.1-------74.48
#-----------------Decision tress-------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier(random_state=0)
print((cross_val_score(clf,X_train, y_train, cv=10)))
# #67.92
# clf = clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# #66.87
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#

#--------------------- grid seach for decision tree
# from sklearn.model_selection import GridSearchCV
# #
# clf = DecisionTreeClassifier()
# param_grid = [{'max_depth':np.arange(1, 21),
#               'min_samples_leaf':[1, 10, 20, 30, 40, 50,60,70,80,90,100]}]
#
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3)
# CV_rfc.fit(X_train, y_train)
#
# r=CV_rfc.best_params_
# print(r)

[0.69942197 0.67919075 0.69565217 0.68313953 0.63953488 0.69387755
 0.6754386  0.70294118 0.67647059 0.64705882]