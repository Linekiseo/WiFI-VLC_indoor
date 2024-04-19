#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/4/19 13:36
# @Author : fanwc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from tensorflow.python.keras.models import load_model
import seaborn as sns


def load_data(csv_path):
    """Loads test data from a CSV file."""
    return pd.read_csv(csv_path)


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using test data."""
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate the classification report for detailed metrics
    print(classification_report(y_test, predicted_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # ROC Curve
    y_prob = predictions[:, 1] if predictions.ndim > 1 else predictions.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def main():
    test_data = load_data('test_data.csv')
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    model = load_model('best_model.h5')
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
