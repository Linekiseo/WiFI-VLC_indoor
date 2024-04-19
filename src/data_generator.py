#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:35
# @Author : fanwc

import numpy as np
import pandas as pd


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

    # Combine data
    combined_data = np.hstack((wifi_data, light_data.reshape(-1, 1)))

    # Convert to DataFrame
    column_names = [f'wifi_{i + 1}' for i in range(num_aps)] + ['light_intensity']
    df = pd.DataFrame(combined_data, columns=column_names)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


# Example usage
if __name__ == "__main__":
    combine_and_save_data(1000, 10)  # Generate 1000 samples, each with data from 10 Wi-Fi APs
