# Spam_Filter

Introduction

We apply classification in our daily lives on a regular basis: organizing things into clearly defined groups, classes, or categories. Classification is helpful because it makes it easier to reason about things and adjust behavior as seen fit. To that end, consider the following: a collection of spam emails vs. normal emails. We all receive them: an email that is sent to a massive number of users at one time, frequently containing scams or phishing content. How do we classify any future incoming emails as spam or not to keep our computers and personal information safe? The answer is to build a spam filter.

Goal

The goal was to create a spam filter that could accurately classify emails based on whether they are spam. The efficacy of two different machine learning models (SVC and Naive Bayes) was also explored, as well as two different vectorizer techniques (CountVectorizer and TfidfVectorizer).

Dataset 

The dataset by https://github.com/SmallLion/Python-Projects/blob/main/Spam-detection/spam.csv mimics the layout of a typical email inbox and includes over 5,000 examples that will be used to train our model.

Implementation

The code uses sklearn as its main library and features two different vectorizer techniques, CountVectorizer and TfidfVectorizer. This code also uses two different machine learning models: SVC and Naive Bayes.

A train-test split method was used to train the email spam detector to recognize and categorize spam emails. The train-test split evaluates the performance of a machine learning algorithm, using it for classification. It takes the dataset and divides it into two separate datasets. The first dataset is used to fit the model and is referred to as the training dataset. For the second dataset, the test dataset, the input element to the model is provided. The dataset contains examples pre-classified into spam and non-spam, using the labels spam and ham, respectively.

CountVectorizer randomly assigns a number to each word in a process called tokenizing. Then, it counts the number of occurrences of words and saves it to cv. At this point, we’ve only assigned a method to cv. Then it randomly assigns a number to each word and counts the number of occurrences of each word saving it to cv.

SVM, the support vector machine algorithm, is a linear model for classification. The idea of SVM is for the algorithm to creates a line which separates the data into classes. 

The TF-IDF Vectorizer creates Tf-IDF values for every word in the dataset. Tf-IDF values are computed in a manner that gives a higher value to words appearing less frequently so that words appearing multiple times do not dominate the less frequent terms.

The Naïve Bayes model, aka MultinomialNB, was fitted to the Tf-IDF vector version of y_train, and the true output labels stored in z_train.

Accuracy of each vectorizer to each model was then measured in order to explore the efficacy of each model on the same data set.

Resources:
1. S. Khan, How to Build a Spam Classifier in 10 Steps (2020),
   
   https://towardsdatascience.com/how-to-build-your-first-spam-classifier-in-10-steps-fdbf5b1b3870  
2. T. Karmali, Spam Classifier in Python from Scratch (2017),
   
   https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73  
3. Dataset: https://github.com/SmallLion/Python-Projects/blob/main/Spam-detection/spam.csv 
4. Full Code: https://github.com/SmallLion/Python-Projects/blob/main/Spam-detection/spam.csv 

