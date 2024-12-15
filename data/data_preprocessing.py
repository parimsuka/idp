# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from data.extract_data import extract_sections_by_elements, combine_elements_data

def load_and_split_data():
    # Load data
    elements_data = extract_sections_by_elements()
    data = combine_elements_data(elements_data)
    # data.head()
    X = data.drop('AOR', axis=1)
    y = data['AOR']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, data



