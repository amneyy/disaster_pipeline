import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'stopwords'])
from nltk.corpus import stopwords

#Model imports
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import sqlite3

#evaluating the model
from sklearn.metrics import classification_report

#save model
import pickle



def load_data(database_filepath):
    """The load_data function read data from the database and returns the dataframe values of df (X, y, category_names)
    
    Parameters:
    database_filepath (db): Filepath of the database
    
    Returns:
    X: message
    y: values of each category names (either 0 or 1)
    category_names: name of the category of df

   """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("select * from my_db", con=conn)
    
    X = df['message']
    y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X,y,category_names


def tokenize(text):
    """The tokenize function takes a text and remove stop words, punctuation at the end of the sentence and apply word_tokenize
    
    Parameters:
    text (string): message to tokenize
    
    Returns:
    list : Tokenized message 

   """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = nltk.word_tokenize(text)
    
    return [w for w in words if w not in stopwords.words('english')]


def build_model():
    """The build function creates and return a ML pipeline
    
    Parameters:
    None
    
    Returns:
    Pipeline

   """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 0.75, 1.0)}
    pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """The evaluate_model function evaluates the accuracy of the model by returning a classification report on the test values
    
    Parameters:
    model (pipeline): Pipeline model created
    X_test: test message
    Y_test: categorization of the X_test message
    category_names: categories 
    
    Returns:
    Classification report of the model

   """
    y_pred = model.predict(X_test)
    
    y_pred_df = pd.DataFrame(y_pred, columns= Y_test.columns)
    for column in Y_test.columns:
        print('{} classification report'.format(column))
        print(classification_report(Y_test[column],y_pred_df[column]))

def save_model(model, model_filepath):
    """The save_model function takes the model and save it into a PKl filename
    
    Parameters:
    model (pipeline): Pipeline model created
    model_filepath: filepath of the model
    
    Returns:
    Pickled Model (.pkl)

   """
    Pkl_Filename = model_filepath
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)
        
    with open(Pkl_Filename, 'rb') as file:  
        Pickled_Model = pickle.load(file)

    return Pickled_Model


def main():
   """The main function loads the data, build and train the model and save it into a pkl file
   Parameters:
   None
   
   Returns:
   None

   """
   if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')

   else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()