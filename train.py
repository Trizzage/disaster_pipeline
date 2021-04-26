# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV

import sqlite3
from sqlalchemy import create_engine

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

import sys

def load_data (database_path):

    ''''
    imports data from a sql database to a pands DataFrame
    Args:
        path_to_database: path to the sql database file
    Returns:
            X: Message data (features for the model)
            y: Categories (targets to predict)
            categories: name of each category of y
            '''
    engine = create_engine('sqlite:///'+database_path)
    df = pd.read_sql_table('disaster_data', con = engine)
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns
    return X, y, categories

def tokenize(text):
    ''''
    function to clean text data by stripping capital letters and removing stopwords
    Args:
    text string
    Returns:
    list of cleaned tokens
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for i in tokens:
        clean_token = lemmatizer.lemmatize(i).lower().strip()
        cleaned_tokens.append(clean_token)

    return cleaned_tokens

def build_model():
    ''''
    function that builds a pipeline and a model using MultiOutputClassifier and Random Forest
    Classifier with GridSearch used to optimize parameters.
    Args:
    None
    Returns:
    cv (model with parameters from GridSearch)
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))])
###note that here my code breaks down. when i went to train the model the first time
###in order to find paramaters to optimize i receieved the following error:
###'Found input variables with inconsistent numbers of samples: [3642, 26216]'
###after many attempts to fix this i have hit a wall, and will complete the rest of my
###functions to the best of my ability as i indend them, once this issue is resolved
    parameters = {
            ###to be filled in}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    return cv

def train(model, X_test, y_test, categories):
    ''''
    function to train and test the model
    Args:
    model, X_test, y_test, categories
    Returns:
    model
    '''
#predicting based on model
    y_pred = model.predict(X_test)
    total_accuracy = 0
    total_f1 = 0

def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    '''
    function to display results of model
    Args:
    cv, y_test, X_test
    Returns:
    f1 of model, confusion matrix, and best parameters for model
    '''
    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)

def export_model(model, filepath):
    ''''
    function to save the model structure and outputs as a pickel file
    Args:
    model created in build model function
    filepath where model will be saved
    Returns:
    pickel file containing the model
    '''
    joblib.dump(model, 'message_model.pkl')

def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    model = display_results(cv, y_test, y_pred)
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
