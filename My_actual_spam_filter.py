# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 13:16:18 2023

@author: kma5
"""
#This code/project categorizes emails based on whether or not they are spam.
#It uses sklearn as its main library.
#The code features two different vectorizer techniques, CountVectorizer and TfidfVectorizer,
#to explore the efficacy of each technique on the same data set.
# This code also uses two different machine learning models: SVC and Naive Bayes.
#to explore the efficacy of each model on the same data set, using each of the vectorizing techniques.

#For more information:
# 1. S. Khan, How to Build a Spam Classifier in 10 Steps (2020),
#    https://towardsdatascience.com/how-to-build-your-first-spam-classifier-in-10-steps-fdbf5b1b3870
# 2. T. Karmali, Spam Classifier in Python from Scratch (2017),
#    https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73
# 3. Dataset: https://github.com/SmallLion/Python-Projects/blob/main/Spam-detection/spam.csv
# 4. Full Code: https://github.com/SmallLion/Python-Projects/blob/main/Spam-detection/spam.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

spam = pd.read_csv('C:\\Users\\kma5\\.conda\\envs\\Spam_Filter\\Spam-detection\\spam.csv')
z = spam['EmailText']
y = spam['Label']
# 20% test, 80% train split
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)


#Count Vectorizer (SVC Model)
cv = CountVectorizer()
features = cv.fit_transform(z_train)

#train a classifier(Count Vectorizer)
model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print("Accuracy (CountVectorizer SVC Model): {}".format(model.score(features_test,y_test)))

#TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(z_train)

#train a classifier(TF-IDF Vectorizer)
model.fit(tfidf_features,y_train)

features_test2 = vectorizer.transform(z_test)
print("Accuracy (TF-IDF Vectorizer SVC Model): {}".format(model.score(features_test2,y_test)))

#Count Vectorizer(Naive Bayes Model)
model2 = MultinomialNB()
model2.fit(features,y_train)

features_test3 = cv.transform(z_test)
print("Accuracy (CountVectorizer Naive Bayes Model): {}".format(model2.score(features_test3,y_test)))


#TF-IDF Vectorizer (Naive Bayes Model)
model2.fit(tfidf_features, y_train)
features_test4 = vectorizer.transform(z_test)

print("Accuracy (TF-IDF Vectorizer Naive Bayes Model): {}".format(model2.score(features_test4,y_test)))
