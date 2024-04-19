#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import numpy as np
import pandas as pd
from model import create_model
from data_preprocessing import preprocess_data, split_data


def train_model(features, labels):
    """Trains the neural network model."""
    # Assume features are already preprocessed and labels are appropriately formatted
    X_train, X_test, y_train, y_test = split_data(features, labels)

    model = create_model(input_shape=X_train.shape[1], num_outputs=1, output_activation='linear')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return history


# Example usage
if __name__ == "__main__":
    # Load data (example path, replace with actual path)
    data = pd.read_csv('processed_data.csv')
    features = preprocess_data(data.drop('label', axis=1))
    labels = data['label'].values

    history = train_model(features, labels)
    print("Model training completed.")

