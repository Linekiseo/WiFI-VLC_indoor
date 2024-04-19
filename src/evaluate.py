#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score


def load_data(csv_path):
    """Loads test data from a CSV file."""
    return pd.read_csv(csv_path)


def evaluate_model(model_path, X_test, y_test):
    """Evaluates the trained model using test data."""
    model = load_model(model_path)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")


# Example usage
if __name__ == "__main__":
    test_data = load_data('test_data.csv')
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    evaluate_model('best_model.h5', X_test, y_test)
