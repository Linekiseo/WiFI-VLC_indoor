#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:43
# @Author : fanwc

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_path):
    """Loads data from a CSV file."""
    return pd.read_csv(csv_path)

def preprocess_data(data):
    """Normalizes the data features."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def split_data(features, labels, test_size=0.2, random_state=42):
    """Splits the data into training and testing datasets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def visualize_processed_data(features, labels):
    """Visualizes preprocessed features with labels."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels)
    plt.title('Scatter Plot of Processed Features')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    data = load_data('data.csv')
    features = preprocess_data(data.drop('light_intensity', axis=1))
    labels = data['light_intensity']
    visualize_processed_data(features, labels)
