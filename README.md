# Coronavirus-Tweet-Sentiment-Analysis

## Abstract
Public sentiments or customer feedbacks are one of the most important aspects in terms of business discussion and government policy making. Sentiment analysis is one of the most effective ways to understand the public/customers emotion, requirement and concern.
COVID-19 one the deadliest pandemic the world has ever witnessed. During this period people faced various concern like oxygen storge, economic instability, lack of medical resources, food scarcity etc. People from different parts of the world shared their conditions through twitter and other social media platform.
Our goal is to analyse the tweet collected from twitter and build a classification model to classify the tweets based on its sentiments. The sentiments may be of various types such as positive, negative, neutral etc.

## Problem Statement

In this challenge we have to build a classification model to predict the sentiment of COVID-19 tweets. The tweets have been pulled from Twitter and manual tagging has been done then. This is a supervised ML classification problem.

## Introduction
The study of sentiment analysis of various tweets during COVID-19 can be helpful for different stakeholders.
For example, government can make use of this information for making new policies by understanding people’s condition and all the challenges they have faced.
Various profit organisation can understand the public requirement and can offer those product and services which have more demand but less supply. For instance, one of the various tweets is talking about scarcity of masks and toilet paper.
NGO’s can also make strategy of how to manage the medical and economical requirement with limited resources.

## Exploratory Data Analysis
A.	Analyse the basic information of the dataset like number of observations and features, Data Type of different features, null values of each feature.

B.	Take a deep dive into “original_tweet” column to understand a brief overview of tweets with different sentiments. Also find the tweets with highest and fewest number of characters.

C.	Finding top 20 location with highest number of tweets.

D.	Top 20 Date with Highest Number of Tweets.

E.	Finding percentage distribution of tweets in terms o
f sentiments.

F.	Finding Top 50 hashtags of all category.

## Training the model

We cannot pass the textual data directly to the ML algorithm. These words need to then be encoded as integers, or floating-point values. We can do it using following methods
### 1)	 Count Vectorizer Method

	Count vectorizer convert a collection of text documents to matrix of integers. Where each integer represents the frequency of the word token in that document.

	This can be implemented in python through CountVectorizer Method. This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.

### 2)	TF-IDF Method

	TF-IDF method represents not only the count of the word token in the document it also reflects how important a word is to a document in collection of corpuses. This can be implemented in python through TfidfVectorizer.

	TF = (Number of times term t appears in a document) / (Number of terms in the document)

	IDF = log(N/n), where, N is the total number of documents and n is the number             of documents the term t has appeared in.
	TF-IDF = TF*IDF

## Building Machine Learning Algorithm
### 1.	Naive Bayes Classifier
Naive Bayes classifiers are a group of class algorithms primarily based on Bayes’ Theorem. It is a generative model which uses probability estimate and likelihood to model the data.
The essential Naive Bayes assumption is that every feature makes an independent and equal contribution to the outcome.
Bayes’ Theorem reveals the chance of an event given the probability of any other event that has already occurred. Bayes’ theorem is said mathematically as the subsequent equation:

P (Y | X) =P (X | Y) P(Y) / P(X)
posterior=prior ∗ likelihood / evidence

This model can be used for both multi class and binary classification.

### 2.	Random Forest Classifier

Every decision tree has an excessive variance, however, while we integrate them all collectively in parallel then the ensuing variance is low as every selection tree receives perfectly trained on that sample data, and as a result, the output doesn’t rely upon one decision tree however on more than one decision trees. In the case of a classification problem, the final output is taken through the use of majority voting. In the case of a regression problem, the final output is the average of output of different tress.

### 3.	Logistic Regression

Logistic regression is essentially a supervised classification algorithm. In a classification problem, the target variable (or output), can take discrete values for a given set of features (or inputs).
Just like Linear regression assumes that the data follows a linear function, Logistic regression models the statistics with the use of the sigmoid function.
Sigmoid function also called as logistic function and is given by:

f(x)= 1/1+e ^(-x)

### 4.	XGBOOST

XGBOOST is an ensemble modelling, approach that tries to construct a robust classifier from the range of weak classifiers.
Gradient Boosting is a famous boosting algorithm. In gradient boosting, every predictor corrects its previous error. XGBOOST is an extension of gradient boosting.
Optimization and Improvement
Regularization:
XGBOOST uses both Lasso and Ridge Regression regularization
Tree Pruning: XGBOOST uses max_depth parameter to restrict overfitting
Cross-validation: XGBOOST implementation comes with a integrated cross- 
            validation method.     
 It also supports parallelization (it can generate the different nodes of tree parallel), can 
  use the hardware    resources efficiently through Cache-Awareness, it can also handle     
  sparse data efficiently.

## Conclusion:  
           
We started the project by loading the dataset collected from twitter. Then to understand the deeper insight of the data we performed EDA on the data. EDA gives us very important insights which help us not only to select right features but also to choose the right ML model for the dataset. The distributions of sentiments, location with highest number of tweets are some of some important insights. After EDA, we extracted and cleaned the important features and pre-process it to a matrix of numbers so that it can be passed to the ML algorithms. Since the target variable have multiple classes, we followed two different approaches for building the models, firstly we pass these pre-processed data directly to multiclass classifier and get the output as multiple classes and secondly, we manipulated the target variable to binary variable and performed binary classifier on it. We applied different ML algorithms such as Naive bays classifier, support vector classifier, Random Forest Classifier, XGBOOST classifier, Logistic Regression etc for both multiclass and binary target variable and evaluated it with different metrics like accuracy score, precision, recall, f1 score etc. We also performed hypermeter tuning to enhance the performance and reduce overfitting of the models. Finally, we got SVC model as best multiclass classifier model with 61.1% test accuracy and logistic regression model as best binary classifier model with 86.5% test accuracy.

