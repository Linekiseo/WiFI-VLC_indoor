#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:43
# @Author : fanwc

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path):
    """Loads data from a CSV file."""
    return pd.read_csv(csv_path)


def preprocess_data(data):
    """Applies preprocessing steps such as normalization to the data."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, :-1])  # Assuming last column as label if present
    return scaled_data


def split_data(features, labels, test_size=0.2, random_state=42):
    """Splits the data into training and testing datasets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    data = load_data('data.csv')
    features = data.drop('light_intensity', axis=1)  # Assuming 'light_intensity' as label, adjust as needed
    labels = data['light_intensity']

    processed_features = preprocess_data(features)
    X_train, X_test, y_train, y_test = split_data(processed_features, labels)
    print("Data split into training and testing sets.")

