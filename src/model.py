#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D



def create_model(input_shape, num_outputs=1, output_activation='sigmoid'):
    """
    Creates a CNN model adjusted to handle sequence data more effectively.

    Parameters:
        input_shape (tuple): The shape of the input data.
        num_outputs (int): The number of output neurons.
        output_activation (str): The activation function of the output layer.

    Returns:
        tf.keras.Model: The constructed neural network model.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_outputs, activation=output_activation)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if output_activation == 'sigmoid' else 'mse',
                  metrics=['accuracy'])
    return model


# Example usage
if __name__ == "__main__":
    model = create_model(input_shape=(10, 1))  # Assume 10 sequential features with 1 channel
    model.summary()
