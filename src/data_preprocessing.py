#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:43
# @Author : fanwc

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def model_wifi_signals(rss_data):
    """Model WiFi signals using Gaussian Processes."""
    X = np.array([[i] for i in range(rss_data.shape[0])])  # Example positional index
    y = rss_data.values
    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1) + Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    return gp

def generate_virtual_reference_points(wifi_data, method='grid-based'):
    """Generate virtual reference points for WiFi data."""
    # Implementation depends on the specific method and data characteristics
    pass

def cluster_light_data(light_data):
    """Cluster light data using KMedoids and DBSCAN."""
    kmedoids = KMedoids(n_clusters=5, random_state=0).fit(light_data)
    labels_kmedoids = kmedoids.labels_
    dbscan = DBSCAN(eps=3, min_samples=2).fit(light_data)
    labels_dbscan = dbscan.labels_
    return labels_kmedoids, labels_dbscan

def preprocess_and_split(data, test_size=0.2, random_state=42):
    """Preprocess data and split into training and testing sets."""
    scaler = StandardScaler()
    feature_columns = [col for col in data.columns if col != 'label']
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    features = data.drop('label', axis=1)
    labels = data['label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(df, file_name):
    """Save DataFrame to CSV file."""
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")

def data_processing_pipeline(wifi_data_path, light_data_path):
    """Process and integrate WiFi and light data, then save processed datasets."""
    wifi_data = pd.read_csv(wifi_data_path)
    light_data = pd.read_csv(light_data_path)

    gp_model = model_wifi_signals(wifi_data)
    generate_virtual_reference_points(wifi_data, method='grid-based')
    labels_kmedoids, labels_dbscan = cluster_light_data(light_data)

    # Assuming merging logic for combined data
    combined_data = pd.DataFrame()  # Placeholder for merged data
    X_train, X_test, y_train, y_test = preprocess_and_split(combined_data)

    # Save processed datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    save_data(train_data, '../data/processed/processed_train_data.csv')
    save_data(test_data, '../data/processed/processed_test_data.csv')

    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

if __name__ == "__main__":
    data_processing_pipeline('../data/raw/wifi_data.csv', '../data/raw/light_data.csv')
