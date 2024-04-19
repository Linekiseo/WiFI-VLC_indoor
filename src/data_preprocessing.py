#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:43
# @Author : fanwc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import GPy


def load_data(csv_path):
    """Loads data from a CSV file."""
    return pd.read_csv(csv_path)


def apply_gaussian_processes(X):
    """Applies Multivariate Gaussian Process Regression to model Wi-Fi signals."""
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X, kernel=kernel)
    model.optimize(messages=True)
    return model.predict(X)


def virtual_reference_point_generation(X):
    """Generates virtual reference points using Gaussian Mixture Model."""
    gmm = GaussianMixture(n_components=5, covariance_type='full')
    gmm.fit(X)
    virtual_points = gmm.sample(n_samples=100)[0]  # Generate 100 virtual points
    return virtual_points


def fusion_clustering(X):
    """Applies fusion of DBSCAN and K-medoids clustering algorithms to the dataset."""
    # DBSCAN to identify core samples
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    core_samples_mask = dbscan.fit_predict(X) != -1

    # K-Medoids on core samples
    core_samples = X[core_samples_mask]
    kmedoids = KMedoids(n_clusters=5, random_state=0).fit(core_samples)
    return kmedoids.cluster_centers_


def preprocess_and_split(data):
    """Normalizes data and splits it into training, testing, and validation sets."""
    scaler = StandardScaler()
    features = scaler.fit_transform(data.drop('label', axis=1))
    labels = data['label'].values

    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    data = load_data('data.csv')
    wifi_data = data.filter(regex='wifi')  # Assuming Wi-Fi data columns are prefixed with 'wifi'
    light_data = data['light_intensity']  # Assuming light intensity is labeled this way

    # Process Wi-Fi data using Gaussian Processes
    wifi_data_modeled = apply_gaussian_processes(wifi_data)

    # Generate virtual reference points
    virtual_points = virtual_reference_point_generation(wifi_data_modeled)

    # Process visible light data using fusion clustering
    light_clusters = fusion_clustering(light_data.values.reshape(-1, 1))

    # Combine all features
    combined_features = np.hstack((wifi_data_modeled, light_clusters, virtual_points))

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(pd.DataFrame(combined_features))
    print("Data preprocessing complete. Data split into train, validation, and test sets.")


if __name__ == "__main__":
    main()
