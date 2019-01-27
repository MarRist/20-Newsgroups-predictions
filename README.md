# 20-Newsgroups-predictions

This repository contains an implementation of several machine learning models used to classify the post to the correct newsgroup. The goal was to evaluate several models and select those with the best test performance. The [20 Newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) is a collection of around 18,000 documents across 20 different newsgroups posts. The total train data consists of 11,314, and the test data consists of 7,532 documents.  

### Descrption of code implementation:

#### **1. Feature Vectorization**

The raw text structures were converted into numerical feature vectors using the vectorization process. The bag-of-words strategy returns a sparse feature matrix X including the word counts, and the frequency-inverse document frequency (tf-idf) approach produces a sparse feature vector that scales down the impact of words that appear frequently in the corpus but are less valuable for the prediction. The formula is:

![eq1](https://latex.codecogs.com/gif.latex?%24%7B%5Cdisplaystyle%20%5Cmathrm%20%7Btfidf%7D%20%28t%2Cd%2CD%29%3D%5Cmathrm%20%7Btf%7D%20%28t%2Cd%29%5Ccdot%20%5Cmathrm%20%7Bidf%7D%20%28t%2CD%29%7D%20%24)

where t is term (feature), d is document, and D is the corpus. For both of the methods, uni-gram tokenization approach was used (i.e. every word was considered as a separate feature). Tf-idf approach was used for building all the classifiers.

#### **2. Feature Extraction**

After the tokenization step, the raw text data was cleaned from numbers, URLs links, punctuation, and stop words. Furthermore, 11,000 features with the highest chi-square score were extracted using [SelectKBest()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html). This was done by computing chi-squared stats between each non-negative feature and each class for features in the range of (1000, 88000) while calculating the train and validation accuracy using the multinomial naive bayes classifier with 10-fold cross-validation. Finally, the feature dimension was reduced to 11,000 features which have reduced the computation time of the classifiers.   

#### **3. Parameter Tuning**

[GridSearchCV()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) procedure was used to search for the best parameters automatically and perform k-fold cross-validation at the same time. 

#### **4. Model Selection**  

Several models were evaluated in order to find the most suitable models for this task, such as: 1) Multinomial and Gaussian Naive Bayes, 2) multi-class logistic regression, 3) support-vector machines, 4) perceptron, 5) multi-layer perceptron, and 6) K-means. The models with the highest accuracy were: multi-layer perceptron and multinomial naive bayes. 
