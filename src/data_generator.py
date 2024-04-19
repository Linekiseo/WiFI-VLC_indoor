#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:35
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend to use
import matplotlib.pyplot as plt
import seaborn as sns


def generate_wifi_data(num_samples, num_aps, rssi_range=(-100, 0)):
    """Generates synthetic Wi-Fi RSSI values."""
    return np.random.randint(rssi_range[0], rssi_range[1], size=(num_samples, num_aps))


def generate_light_data(num_samples, light_range=(100, 1000)):
    """Generates synthetic light intensity data."""
    return np.random.randint(light_range[0], light_range[1], size=(num_samples,))


def generate_labels(num_samples, imbalance_ratio=0.5):
    """Generates binary labels with a specified imbalance ratio."""
    # imbalance_ratio is the proportion of the minority class
    num_minority = int(num_samples * imbalance_ratio)
    num_majority = num_samples - num_minority
    return np.concatenate([np.ones(num_minority), np.zeros(num_majority)])


def combine_and_save_data(num_samples, num_aps, imbalance_ratio, output_path='data.csv'):
    """Generates Wi-Fi and light data, combines them with labels, and saves to CSV."""
    wifi_data = generate_wifi_data(num_samples, num_aps)
    light_data = generate_light_data(num_samples)
    labels = generate_labels(num_samples, imbalance_ratio)

    # Combine data
    combined_data = np.hstack((wifi_data, light_data.reshape(-1, 1), labels.reshape(-1, 1)))
    column_names = [f'wifi_{i + 1}' for i in range(num_aps)] + ['light_intensity', 'label']
    df = pd.DataFrame(combined_data, columns=column_names)

    df.to_csv(output_path, index=False)
    return df


def visualize_data(df):
    """Visualizes distributions of generated data and label balance."""

    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=df)
    plt.title('Distribution of Labels')
    plt.show()

    sns.pairplot(df.drop('label', axis=1))
    plt.suptitle('Pairplot of Features')
    plt.show()


if __name__ == "__main__":
    df = combine_and_save_data(100000, 10, 0.2, 'data.csv')  # 20% minority class
    visualize_data(df)
