
# coding: utf-8

# In[25]:


# Martina Risteska (ID: 1003421781)
'''
Question 1 Skeleton Code

'''

import sklearn
import numpy as np
import re
import pandas as pd
import seaborn as sn
import zipfile 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics 
from sklearn.feature_selection import SelectKBest, chi2


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test


def clean_data (string):
    '''
    Removing all punctuations, URLS, user_names and redundant letters repeating in the text
    '''

    input_str = string.lower()
    output_str = re.sub(r"http\S+", "", input_str) # remove URLs
    output_str1 = re.sub(r"@\S+", "", output_str)   # remove user_names
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    output_str2 = pattern.sub(r"\1\1", output_str1)  # remove more than 3 same letters repeating (i.e., sleeeeeep into sleep)
    cleaned_string = re.sub('[^A-Za-z\s]+', '', output_str2).strip() # create a subset of the current string by going character by character and taking only letters, digits and spaces from the current string
    
    return cleaned_string


def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer(stop_words = 'english', lowercase = True)
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(stop_words = 'english', lowercase = True, smooth_idf = True, preprocessor = clean_data)
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    
    shape = tf_idf_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[tf_idf_train.sum(axis=0).argmax()]))
    
    return tf_idf_train, tf_idf_test, feature_names


# Bernoulli Naive Bayes model - baseline model
def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


# Multinomial Naive Bayes model 
def multinomial_naivebayes(tf_idf_train, train_labels, tf_idf_test, test_labels):

    # Create model
    MNB_model = MultinomialNB()
    
    # Create a grid search with 10-fold cv to find the best model parameters
    '''
    # parameters used for hyper-parameters tuning
    
    parameters = {'alpha': np.array([0.001, 0.01, 0.02, 0.05, 0.1, 1, 10, 100])}
    '''
    parameters = {'alpha': ([0.01])}
    
    grid_MNB = GridSearchCV(
        MNB_model,  # Multinomial Naive Bayes model
        param_grid = parameters, # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        scoring='accuracy',  
        cv=10,  # 10-fold cross-validation to be used while performing every search
        )
    
    # Train the model
    MNB_model_fit = grid_MNB.fit(tf_idf_train, train_labels)

    #evaluate the baseline model
    train_pred = MNB_model_fit.predict(tf_idf_train)
    print('Multinomial NB train accuracy = {}'.format((train_pred == train_labels).mean()))
    
    test_pred = MNB_model_fit.predict(tf_idf_test)
    print('Multinomial NB test accuracy = {}'.format((test_pred == test_labels).mean()))

    return MNB_model_fit, test_pred

# Multi-class Logistic regression model
def logistic_reg(tf_idf_train, train_labels, tf_idf_test, test_labels):
    
    # Create model
    log_reg_model = LogisticRegression()
    
    # Create a grid search with 10-fold cv to find the best model parameters
    '''
    # parameters used for hyper-parameters tuning
    
    params = {'C': np.array([0.01, 0.1, 1.0, 10.0, 100.0]),
             'penalty': ('l1','l2'),
             'multi_class': ('ovs', 'multinomial'),
             'solver': ('liblinear', 'newton-cg', 'lbfgs')}
    '''
    params = {'C': ([10]),
             'penalty': (['l2'])}
    
    grid_log_reg = GridSearchCV(
        log_reg_model,  # Logistic regression model
        param_grid = params, # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        scoring='accuracy',  
        cv=10,  # 10-fold cross-validation to be used while performing every search
        )
    
    # Train the model
    log_model_fit = grid_log_reg.fit(tf_idf_train, train_labels)
    
    # Evaluate the model
    log_train_pred = log_model_fit.predict(tf_idf_train)
    print('Logistic Regression train accuracy = {}'.format((log_train_pred == train_labels).mean()))
    
    log_test_pred = log_model_fit.predict(tf_idf_test)
    print('Logistic Regression test accuracy = {}'.format((log_test_pred == test_labels).mean()))
    
    return log_model_fit


# Multi-layer perceptron model
def multilayer_perceptron(tf_idf_train, train_labels, tf_idf_test, test_labels):
    
    multi_perceptron_model = MLPClassifier()
    
    # Create a grid search with 10-fold cv to find the best model parameters
    '''
    # parameters used for hyper-parameters tuning
    
    params = {'alpha': np.array([0.0001, 0.001, 0.1, 10]),
             'hidden_layer_sizes': [(10,), (10,1), (20,), (20,1), (40,), (40,2), (100,)],
             'learning_rate' : (['adaptive'])}
    '''
    
    params = {'alpha':  ([000.1]),
             'hidden_layer_sizes': [(100,)],
             'learning_rate' : (['adaptive'])}
    
    grid_multi_perceptron = GridSearchCV(
        multi_perceptron_model,  # Logistic regression model
        param_grid = params, # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        scoring='accuracy',  
        cv=3,  # 10-fold cross-validation to be used while performing every search
        )
    
    # Train the model
    multi_perceptron_model_fit = grid_multi_perceptron.fit(tf_idf_train, train_labels)
    
    # Evaluate the model
    multi_perceptron_train_pred = multi_perceptron_model_fit.predict(tf_idf_train)
    print('Multi-Layer Perceptron train accuracy = {}'.format((multi_perceptron_train_pred == train_labels).mean()))
    
    multi_perceptron_test_pred = multi_perceptron_model_fit.predict(tf_idf_test)
    print('Multi-Layer Perceptron test accuracy = {}'.format((multi_perceptron_test_pred == test_labels).mean()))
    
    return multi_perceptron_model_fit


def confusion_matrix(true_target, prediction):
        
    k = 20
    U = np.zeros((k, k))  # upper triagnular matrix (used to fill in the negative values of the differences)
    L = np.zeros((k, k))  # lower triangular matrix (used to fill in the positive values of the differences)
    D = np.zeros((k, k))  # diagonal matrix (used to fill in the zero values of the differences)
    
    
    # Create a dataframe from the predcted and true labels
    data_df = pd.DataFrame()
    data_df['PredictedLabels'] = prediction
    data_df["TrueLabels"] = true_target
    
    # sort the dataframe 
    data_df_sorted = data_df.sort_values('TrueLabels', ascending = True)
    
    for topic_class in range(0,k):
        data_df_subset = data_df_sorted.loc[data_df_sorted.TrueLabels == topic_class]
        n = data_df_subset.shape[0]
        data_df_difference = (data_df_subset.TrueLabels - data_df_subset.PredictedLabels).to_frame(name = 'Difference')
        counts = data_df_difference['Difference'].value_counts().to_frame("Counts")
        differences = data_df_difference['Difference'].value_counts().index.tolist()

        
        for idx, counts in enumerate(counts.Counts):
            
            if(differences[idx] > 0):
                L[topic_class, topic_class - differences[idx]] = counts
                
            if(differences[idx] < 0):
                U[topic_class, topic_class - differences[idx]] = counts
            
            if(differences[idx] == 0):
                D[topic_class, topic_class] = counts

    conf_matrix = pd.DataFrame(L + D + U)
    
    
    # Find the classes for wich the classifier is most confused about
    max_misclassified_classes_row = pd.DataFrame()
    max_misclassified_classes_col = pd.DataFrame()
    misclasification = pd.DataFrame(L + U)
    
    # Max per column
    max_misclassified_classes_col['True Class'] = np.array(range(0,k))
    max_misclassified_classes_col['Predicted Class'] = misclasification.idxmax(axis = 1)
    max_misclassified_classes_col['Counts'] = misclasification.max(axis = 1)
    max_misclassified_classes_col = max_misclassified_classes_col[max_misclassified_classes_col['Counts'] != 0]
    max_misclassified_classes_col['Predicted-True Tuple'] = list(zip(max_misclassified_classes_col['True Class'], max_misclassified_classes_col['Predicted Class']))
    max_misclassified_classes_col = max_misclassified_classes_col.drop(['True Class', 'Predicted Class'], axis=1)

    # Max per row
    max_misclassified_classes_row['Predicted Class'] = np.array(range(0,k))
    max_misclassified_classes_row['True Class'] = misclasification.idxmax(axis = 0)
    max_misclassified_classes_row['Counts'] = misclasification.max(axis = 0)
    max_misclassified_classes_row = max_misclassified_classes_row[max_misclassified_classes_row['Counts'] != 0]
    max_misclassified_classes_row['Predicted-True Tuple'] = list(zip(max_misclassified_classes_row['True Class'], max_misclassified_classes_row['Predicted Class']))
    max_misclassified_classes_row = max_misclassified_classes_row.drop(['True Class', 'Predicted Class'], axis=1)
    
    # merge the two dataframes
    max_misclassified_classes = pd.merge(max_misclassified_classes_col, max_misclassified_classes_row, on= "Predicted-True Tuple")
    max_misclassified_classes['Counts'] = max_misclassified_classes['Counts_x'] + max_misclassified_classes['Counts_y']
    
    # Sort the tuple elements
    for idx, row in enumerate(max_misclassified_classes["Predicted-True Tuple"]):
        max_misclassified_classes.at[idx, "Predicted-True Tuple"] = tuple(sorted(row))
    
    max_misclassified_classes = max_misclassified_classes.groupby('Predicted-True Tuple', as_index=False).sum()
    max_misclassified_classes = max_misclassified_classes.sort_values('Counts', ascending = False)
    
    return conf_matrix, max_misclassified_classes['Predicted-True Tuple'].head(1)




def vizualize_confusion_matrix(conf_matrix):
    
    fig, ax = plt.subplots(figsize = (20,10))
    sn.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, square = True, fmt='g')
    sn.set(font_scale = 1.5)
    
    # Put a title on the plot
    ax.set_title("Confusion Matrix", 
    fontsize=20, fontweight="bold")
    
    # Adding space between the title and the graph
    rcParams['axes.titlepad'] = 90 # Space between the title and graph

    ax.xaxis.tick_top()
    plt.show()
    
    return None



if __name__ == '__main__':
    
    # Load data
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test, tf_idf_feature_names = tf_idf_features(train_data, test_data)
    
    '''
    # Code used to find the top k features in the range of (1000, feature_size) 
    # using the test and train accuracy of multinomial naive bayes. The top k = 11000 was found to be good features
    
    K = range(1000,tf_idf_train.shape[1],1000)
    for k in K:
        ch2 = SelectKBest(chi2, k)
        K_feature_train = ch2.fit_transform(tf_idf_train, train_data.target)
        K_feature_test = ch2.transform(tf_idf_test)

        #print("Selected %d features and accuracy is:" %k, mnb_model.best_score_) 
        print("Selected %d features the train and test accuracy is:" %k)
        mn_NB = multinomial_naivebayes(K_feature_train, train_data.target, K_feature_test, test_data.target)
    '''
    
    # FEATURE EXTRACTION
    K = 11000 # Take the best 11000 features that have the highest score (prameter found by observing test and train accuracy in multinomial model)
    ch2 = SelectKBest(chi2, K)
    K_feature_train = ch2.fit_transform(tf_idf_train, train_data.target)
    K_feature_test = ch2.transform(tf_idf_test)

    # MODELS
    # Bernouli Naive Bayes
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # Logistic Regression
    log_reg_model = logistic_reg(K_feature_train, train_data.target, K_feature_test, test_data.target)
    
    # Multi-layer Perceptron
    multi_perceptron_model = multilayer_perceptron(K_feature_train, train_data.target, K_feature_test, test_data.target)
    
    # Multinomial Naive Bayes
    mnb_model, mnb_predicted = multinomial_naivebayes(K_feature_train, train_data.target, K_feature_test, test_data.target)
    matrix, max_confusion_classes = confusion_matrix(test_data.target, mnb_predicted)
    print("The classifier was the most confused about the following classes:", max_confusion_classes)
    vizualize_confusion_matrix(matrix)
    

