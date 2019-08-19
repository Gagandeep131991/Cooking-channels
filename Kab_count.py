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

data=pd.read_csv("kabitakitchen.csv",encoding="latin-1")

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
#print(data['word_count'])
#print(data[['commentText','word_count']].head())
# #
#Count the number of characters
data['char_count'] = data['commentText'].str.len() #It includes spaces too
#print(data[['commentText','char_count']].head())
# #
#Average words
def avg_word(sentence):
  words = sentence.split()
  try:
    return (sum(len(word) for word in words)/len(words))
  except:
      print(sentence)

data['avg_word'] = data['commentText'].apply(lambda x: avg_word(x))
#print(data[['commentText','avg_word']])
# #
#
# #
#count the numeric values in each comment
data['numerics'] = data['commentText'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#print(data[['commentText','numerics']].head())
#
#
#
data.to_csv('kabita_preprocessing.csv', index=False)
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
# import numpy as np # linear algebra
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# from subprocess import check_output
# from wordcloud import WordCloud, STOPWORDS
#
# #mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
# mpl.rcParams['font.size']=12                #10
# mpl.rcParams['savefig.dpi']=100             #72
# mpl.rcParams['figure.subplot.bottom']=.1
#
#
# stopwords = set(STOPWORDS)
#
# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stopwords,
#                           max_words=200,
#                           max_font_size=40,
#                           random_state=42
#                          ).generate(str(data['commentText']))
#
# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)
# sentences = []
#
# for text in data.commentText:
# 	sentences.append(text)
# #print(sentences)
# #
# vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(sentences).toarray()

#X_train = vectorizer.fit_transform(sentences).toarray()
#print(X)
#
# #X = X.todense()
#
#X_train, X_test, y_train, y_test = train_test_split(X, data['Labels'], test_size=0.3, random_state=5)
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
# #
# print((cross_val_score(classifier,X_train, y_train, cv=10)))
# # #71.21
#[0.71676301 0.74277457 0.69855072 0.70639535 0.73837209 0.71428571
# 0.70760234 0.71764706 0.67647059 0.70294118]
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
# # #70.61
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# #66.19
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 62.62
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#70.08
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#70.61
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#73.32

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
#
# clf = MultinomialNB(alpha=0.1)
# print((cross_val_score(clf,X_train, y_train, cv=10)))
# ##69.47
#[0.67919075 0.70520231 0.73623188 0.65697674 0.70639535 0.72594752
# 0.69298246 0.7        0.66176471 0.68235294]

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Multinomial naive bayes")
# print(accuracy_score(y_test, y_pred))
# #69.31
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


#
# bnb = BernoulliNB(alpha=0.1)
# print((cross_val_score(bnb,X_train, y_train, cv=10)))
# # #67.51
# [0.6734104  0.66763006 0.72753623 0.62790698 0.68895349 0.67638484
#  0.69298246 0.67941176 0.67352941 0.64411765]

# bnb.fit(X_train, y_train)
# y_pred=bnb.predict(X_test)
# print("Bernoulli nayive bayes")
# print(accuracy_score(y_test, y_pred))
# #66.53
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#
# Bernoulli nayive bayes
# 0.6653061224489796
# mcc 0.6135920129623145
# F1 score: 0.6528338181473746
# Recall: 0.6653061224489796
# Precision: 0.6646365335775504
#-----------------Gaussian Naive Bayes--------------------
# from sklearn.naive_bayes import GaussianNB
#
# gnb= GaussianNB()
# print((cross_val_score(gnb,X_train, y_train, cv=10)))#48.90
# [0.43930636 0.50289017 0.48405797 0.4505814  0.47383721 0.52478134
#  0.52631579 0.50294118 0.50882353 0.47647059]

# gnb.fit(X_train, y_train)
# y_pred=gnb.predict(X_test)
# print("Gaussian naive bayes")
# print(accuracy_score(y_test, y_pred))
# # #47.61
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#
# #Gaussian naive bayes
# 0.47619047619047616
# mcc 0.39343897851678034
# F1 score: 0.4647582680640981
# Recall: 0.47619047619047616
# Precision: 0.4884355188580142

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
# # #
# model = LogisticRegression(penalty='l2',C=10, dual=False, multi_class='multinomial', solver='newton-cg')
# print((cross_val_score(model,X_train, y_train, cv=10)))#74.71
# [0.72543353 0.76589595 0.77101449 0.72383721 0.76744186 0.76093294
#  0.75438596 0.77352941 0.71764706 0.71176471]
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Logistic Regression")
# print(accuracy_score(y_test, y_pred))
# 73.67
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
#0.1


#-------------------Linear Support Vector Machine------------------

# from sklearn import svm
# #
# clfr = svm.SVC(C=1, decision_function_shape='ovr', kernel='linear',
#       probability=False, random_state=None, shrinking=True,
#       tol=0.001, verbose=False)
# #
# print((cross_val_score(clfr,X_train, y_train, cv=10)))#71.45(0.1)...74.60
# [0.73988439 0.7716763  0.75362319 0.73546512 0.73255814 0.75510204
#  0.74561404 0.77058824 0.74117647 0.71470588]
# # clfr.fit(X_train, y_train)
# y_pred = clfr.predict(X_test)
# # # # #
# from sklearn import metrics
# # # # #
# # # Model Accuracy: how often is the classifier correct?
# print("SVM Linear:",metrics.accuracy_score(y_test, y_pred))
# # 70.54(0.1)----74.55
#
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98


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
# [0.66473988 0.69364162 0.66666667 0.62209302 0.66569767 0.64431487
#  0.64912281 0.67941176 0.59705882 0.64411765]
# #65.07
# # #Train the model using the training sets
# clfpol.fit(X_train, y_train)
# #
# # #Predict the response for test dataset
# y_pred = clfpol.predict(X_test)
# print("SVM Polynomial:",metrics.accuracy_score(y_test, y_pred))
# #64.14
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98

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




# #--------------SVM(Gaussian)---------------------------
# clfgs = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print((cross_val_score(clfgs,X_train, y_train, cv=10)))
# #73.69
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
# #74.01
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98
#

#C=1000,gamma=0.001-----74.01
#  1000,      0.01-------73.12
#  100,       0.001-----71.70
#1000,       0.1      71.90
#-----------------Decision tress-------------------

# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# clf = DecisionTreeClassifier(random_state=0)
# print((cross_val_score(clf,X_train, y_train, cv=10)))
# #68.88
# clf = clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# #67.82
# print("mcc", matthews_corrcoef(y_test, y_pred))
# # 64.32
# print ('F1 score:', f1_score(y_test, y_pred,average='weighted'))#68.67
# print ('Recall:', recall_score(y_test, y_pred,average='weighted'))#69.31
# print ('Precision:', precision_score(y_test, y_pred, average='weighted'))#68.98


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


# [0.73988439 0.76300578 0.74202899 0.72965116 0.73255814 0.74344023
#  0.74561404 0.76764706 0.69705882 0.70882353]
# [0.71676301 0.69075145 0.68985507 0.69476744 0.68313953 0.72886297
#  0.64912281 0.69117647 0.66470588 0.67941176]