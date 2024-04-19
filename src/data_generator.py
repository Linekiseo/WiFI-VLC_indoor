#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:35
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_wifi_data(num_samples, num_aps, rssi_range=(-100, 0)):
    """Generates synthetic Wi-Fi RSSI values."""
    return np.random.randint(rssi_range[0], rssi_range[1], size=(num_samples, num_aps))


def generate_light_data(num_samples, light_range=(100, 1000)):
    """Generates synthetic light intensity data."""
    return np.random.randint(light_range[0], light_range[1], size=(num_samples,))


def combine_and_save_data(num_samples, num_aps, output_path='data.csv'):
    """Generates Wi-Fi and light data, combines them, and saves to CSV."""
    wifi_data = generate_wifi_data(num_samples, num_aps)
    light_data = generate_light_data(num_samples)

    combined_data = np.hstack((wifi_data, light_data.reshape(-1, 1)))
    column_names = [f'wifi_{i + 1}' for i in range(num_aps)] + ['light_intensity']
    df = pd.DataFrame(combined_data, columns=column_names)

    df.to_csv(output_path, index=False)
    return df


def visualize_data(df):
    """Visualizes distributions of generated data."""
    plt.figure(figsize=(12, 6))
    for i in range(df.shape[1] - 1):
        sns.kdeplot(df.iloc[:, i], label=f'WiFi AP {i + 1}')
    plt.title('Distribution of WiFi RSSI Values')
    plt.legend()
    plt.show()

    sns.histplot(df['light_intensity'], kde=True, color='green')
    plt.title('Distribution of Light Intensity')
    plt.show()


if __name__ == "__main__":
    df = combine_and_save_data(1000, 10, 'data.csv')
    visualize_data(df)
