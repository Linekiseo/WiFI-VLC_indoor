#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:35
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

def log_distance_path_loss(x, y, ap_positions, ref_distance=1.0, path_loss_exp=2.7, ref_signal=-40, noise_level=2):
    """Simulates WiFi signal using Log-Distance Path Loss Model."""
    signal_strength = []
    for ap in ap_positions:
        distance = max(ref_distance, np.sqrt((x - ap[0])**2 + (y - ap[1])**2))
        loss = ref_signal - 10 * path_loss_exp * np.log10(distance / ref_distance)
        signal_strength.append(loss + np.random.normal(0, noise_level))
    return signal_strength

def zone_based_labeling(x, y):
    """Assign labels based on predefined functional zones in the environment."""
    if x < 50 and y < 50:
        return 'Common Area'
    elif x >= 50 and y < 50:
        return 'Office'
    elif x < 50 and y >= 50:
        return 'Hallway'
    else:
        return 'Meeting Room'

def intersects(point, light, obstacle):
    """Check if the line from point to light intersects with an obstacle."""
    # Simplified example, assumes obstacle is defined by bottom-left and top-right coordinates
    obs_bottom_left, obs_top_right = obstacle
    # Implement line-rectangle intersection logic here, potentially using line equations and bounding box checks
    return False  # Placeholder return


def light_intensity_with_obstacles(x, y, light_positions, obstacles, max_intensity=800, falloff=30):
    """Calculates light intensity considering obstacles which may block or reflect light."""
    intensity = 0
    for light in light_positions:
        direct_line = True
        for obs in obstacles:
            if intersects((x, y), light, obs):
                direct_line = False
                break
        distance = np.sqrt((x - light[0])**2 + (y - light[1])**2)
        if direct_line:
            intensity += max_intensity / (1 + (distance / falloff)**2)
        else:
            intensity += (max_intensity / (1 + (distance / falloff)**2)) * 0.5  # Reduced by obstacles
    return intensity

def simulate_environment(num_samples, area_size=(100, 100), num_aps=4, num_lights=3):
    ap_positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])) for _ in range(num_aps)]
    light_positions = [(np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])) for _ in range(num_lights)]
    obstacles = [((50, 50), (60, 60)), ((20, 80), (30, 90))]  # Example obstacles

    records = []
    for _ in range(num_samples):
        x, y = np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])
        wifi_signals = log_distance_path_loss(x, y, ap_positions)
        light_intensity = light_intensity_with_obstacles(x, y, light_positions, obstacles)
        label = zone_based_labeling(x, y)  # Function to determine the zone based label
        records.append([x, y] + wifi_signals + [light_intensity, label])

    columns = ['x', 'y'] + [f'wifi_signal_{i+1}' for i in range(num_aps)] + ['light_intensity', 'label']
    return pd.DataFrame(records, columns=columns)

def visualize_data(data):
    # Convert categorical labels to a categorical type if they aren't already
    if not isinstance(data['label'].dtype, pd.CategoricalDtype):
        data['label'] = data['label'].astype('category')

    # Assign a number to each category label (also supports non-numeric labels)
    label_numbers = data['label'].cat.codes

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data['x'], data['y'], c=label_numbers, cmap='viridis', alpha=0.6, edgecolors='w', linewidths=0.5)

    # Create a colorbar with the correct label names
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Zone Label')
    colorbar.set_ticks(range(len(data['label'].cat.categories)))
    colorbar.set_ticklabels(data['label'].cat.categories)

    plt.title('Spatial Distribution of Functional Zones')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)  # Optional: Adds grid for better visualization
    plt.show()


if __name__ == "__main__":
    data = simulate_environment(100000)
    data.to_csv('enhanced_simulated_indoor_data.csv', index=False)
    visualize_data(data)

