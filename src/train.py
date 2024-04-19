#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import numpy as np
import pandas as pd
from model import create_model
from data_preprocessing import preprocess_data, split_data, load_data
import matplotlib.pyplot as plt


def train_model(features, labels):
    """Trains the neural network model and visualizes the training process."""
    X_train, X_test, y_train, y_test = split_data(features, labels)
    model = create_model(input_shape=X_train.shape[1], num_outputs=1, output_activation='linear')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return history


def plot_training_history(history):
    """Plots training and validation loss and accuracy."""
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

def main():
    data = load_data('data.csv')
    features = preprocess_data(data.drop('light_intensity', axis=1).values)
    labels = data['light_intensity'].values

    history = train_model(features, labels)
    plot_training_history(history)

if __name__ == "__main__":
    main()

