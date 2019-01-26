# 20-Newsgroups-predictions

This repository contains an implementation of several machine learning models used to classify the post to the correct newsgroup. The goal was to evaluate several models and select those with the best test performance. The 20 Newsgroups data set is a collection of around 18,000 documents across 20 different newsgroups posts. The total train data consists of 11,314, and the test data consists of 7,532 documents.  

**1. Feature Vectorization**

To work with text data there are two methods that converts the raw text structures into numerical feature vectors using the vectorization process. The bag-of-words strategy returns a spare feature matrix X including the word counts, and the frequency-inverse document frequency (tf-idf) approach produces a sparse feature vector that scales down the impact of words that appear frequently in the corpus but are less valuable for the prediction. The formula is:

![eq1](https://latex.codecogs.com/gif.latex?%24%7B%5Cdisplaystyle%20%5Cmathrm%20%7Btfidf%7D%20%28t%2Cd%2CD%29%3D%5Cmathrm%20%7Btf%7D%20%28t%2Cd%29%5Ccdot%20%5Cmathrm%20%7Bidf%7D%20%28t%2CD%29%7D%20%24)

where t is term (feature), d is document, and D is the corpus. For both of the methods, uni-gram tokenization approach was used (i.e. every word was considered as a separate feature). Tf-idf approach was used for building all the classifiers.

**2. Feature Extraction**

Feature cleaning is an important step, as text data is usually highly redundant which result in a lot of invaluable features to appear in the corpus. This is aparent from the feature dimension being much higher than the size of the train data (101,631 vs. 11,314). For that purpose, the raw text data after the tokenization, was cleaned from numbers, URLs links, and punctuation, all for the purpose of improving the classifiers predictive power. Moreover, words such as "the", known as stop words were also removed from the corpus. After data cleaning the feature dimension of 101,631 was reduced to 88,108 features. Furthermore, 11,000 features with the highest chi-square score were extracted using the SelectKBest() procedure in scikit-learn. This was done by computing chi-squared stats between each non-negative feature and each class for features in the range of (1000, 88000) while calculating the train and validation accuracy using the multinomial naive bayes classifier with 10-fold cross-validation. Finally, the feature dimension was reduced to 11,000 features which have reduced the computation time of the classifiers. It is worth noting that such feature cleaning did not improve the classifiers performance since tf-idf approach does a good job at marking such features as invaluable by assigning a low score. Several remedies may be used to extract better features such as word stemming, synonym finding, using pairs of words, or better approach for text extraction such as latent Dirichlet allocation (LDA).    

**3. Parameter Tuning**

GridSearchCV() procedure from scikit-learn python library was used to search for the best parameters automatically and perform k-fold cross-validation at the same time. The values of the parameters were pre-defined and vary from the type of the classifier. For all classifiers 'scoring' accuracy was used to find the best model parameters, and to return the refitted estimator using the best found parameters on the train set.  

**4. Model Selection**  Several models were evaluated in order to find the most suitable models for this task, such as: 1) Multinomial and Gaussian Naive Bayes, 2) multi-class logistic regression, 3) support-vector machines, 4) perceptron, 5) multi-layer perceptron, and 6) K-means. The models with the highest accuracy were: multi-class logistic regression, multi-layer perceptron and multinomial naive bayes. Since these models had the best performance, were further considered for the text classification task. 
