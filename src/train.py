#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend to use
import matplotlib.pyplot as plt

def load_data():
    # This function should load your data properly formatted for training
    # Example: return features, labels
    return np.random.random((100, 10, 1)), np.random.randint(2, size=(100,))

def train_and_evaluate():
    X, y = load_data()
    model = create_model(input_shape=X.shape[1:])

    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    history = model.fit(X, y, epochs=50, validation_split=0.2, callbacks=[checkpoint, early_stop], batch_size=32)

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    train_and_evaluate()

