#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:43
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def model_wifi_signals(rss_data):
    print("Modeling WiFi signals...")
    X = np.array([[i] for i in range(rss_data.shape[0])])
    y = rss_data.values
    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1) + Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    return gp

def generate_virtual_reference_points(wifi_data, method='grid-based'):
    print("Generating virtual reference points...")
    pass

def cluster_light_data(light_data):
    print("Clustering light data...")
    kmedoids = KMedoids(n_clusters=5, random_state=0).fit(light_data)
    labels_kmedoids = kmedoids.labels_
    dbscan = DBSCAN(eps=3, min_samples=2).fit(light_data)
    labels_dbscan = dbscan.labels_
    return labels_kmedoids, labels_dbscan

def preprocess_and_split(data, test_size=0.2, random_state=42):
    print("Preprocessing and splitting data...")
    scaler = StandardScaler()
    feature_columns = [col for col in data.columns if col != 'label']
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    features = data.drop('label', axis=1)
    labels = data['label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(df, file_name):
    print(f"Saving data to {file_name}...")
    df.to_csv(file_name, index=False)

def visualize_data(features, labels):
    print("Visualizing data...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels)
    plt.title('Feature Distribution After Processing')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Label')
    plt.show()

def data_processing_pipeline(wifi_data_path, light_data_path):
    wifi_data = pd.read_csv(wifi_data_path)
    light_data = pd.read_csv(light_data_path)

    gp_model = model_wifi_signals(wifi_data)
    generate_virtual_reference_points(wifi_data, method='grid-based')
    labels_kmedoids, labels_dbscan = cluster_light_data(light_data)

    combined_data = pd.DataFrame()  # Placeholder for combined data after merging
    X_train, X_test, y_train, y_test = preprocess_and_split(combined_data)

    save_data(pd.concat([X_train, y_train], axis=1), 'processed_train_data.csv')
    save_data(pd.concat([X_test, y_test], axis=1), 'processed_test_data.csv')

    visualize_data(X_train.values, y_train.values)

    print("Data processing complete. Training and testing data are ready for use.")

if __name__ == "__main__":
    data_processing_pipeline('../data/raw/wifi_data.csv', '../data/raw/light_data.csv')
