#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_shape, num_outputs=1, output_activation='linear'):
    """Creates a simple neural network model for the given input shape."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_outputs, activation=output_activation)
    ])

    model.compile(optimizer='adam',
                  loss='mse' if output_activation == 'linear' else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    model = create_model(input_shape=10)  # Assume 10 features from preprocessed data
    model.summary()
