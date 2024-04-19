#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 16:42
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_wifi_signal(x, y, ap_positions, noise_level=5):
    """Generates WiFi signal strength based on distance to the nearest access point and adds noise."""
    signal_strength = []
    for pos in ap_positions:
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        strength = -30 - 20 * np.log10(distance)  # Simplified path loss model
        strength += np.random.normal(0, noise_level)  # Adding Gaussian noise
        signal_strength.append(strength)
    return signal_strength

def generate_light_intensity(x, y, light_positions, max_intensity=1000, falloff=50):
    """Generates light intensity based on inverse square law relative to nearest light source."""
    intensity = 0
    for light in light_positions:
        distance = np.sqrt((x - light[0])**2 + (y - light[1])**2)
        intensity += max_intensity / (1 + (distance / falloff)**2)
    return intensity

def simulate_environment(num_samples, area_size=(100, 100), num_aps=3, num_lights=2):
    ap_positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])) for _ in range(num_aps)]
    light_positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])) for _ in range(num_lights)]

    wifi_records = []
    light_records = []
    for _ in range(num_samples):
        x, y = np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])
        wifi_signals = generate_wifi_signal(x, y, ap_positions)
        light_intensity = generate_light_intensity(x, y, light_positions)
        wifi_records.append([x, y] + wifi_signals)
        light_records.append([x, y, light_intensity])

    wifi_columns = ['x', 'y'] + [f'wifi_signal_{i+1}' for i in range(num_aps)]
    light_columns = ['x', 'y', 'light_intensity']

    df_wifi = pd.DataFrame(wifi_records, columns=wifi_columns)
    df_light = pd.DataFrame(light_records, columns=light_columns)

    return df_wifi, df_light

def visualize_data(df, title, intensity_label):
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df['x'], df['y'], c=df[intensity_label], cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label=intensity_label)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

def save_data(df_wifi, df_light, wifi_path='../data/raw/wifi_data.csv', light_path='../data/raw/light_data.csv'):
    df_wifi.to_csv(wifi_path, index=False)
    df_light.to_csv(light_path, index=False)
    print(f"WiFi data saved to {wifi_path}")
    print(f"Light data saved to {light_path}")

if __name__ == "__main__":
    df_wifi, df_light = simulate_environment(10000)
    save_data(df_wifi, df_light)
    visualize_data(df_wifi, 'WiFi Signal Strength Distribution', 'wifi_signal_1')
    visualize_data(df_light, 'Light Intensity Distribution', 'light_intensity')

