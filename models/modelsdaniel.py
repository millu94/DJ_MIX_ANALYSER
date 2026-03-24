import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import os

def load_data(filepath="../datasets/processed/features.csv"):
    if not os.path.exists(filepath):
        # try fallback path
        filepath = "datasets/processed/features.csv"
    df = pd.read_csv(filepath)
    # Exclude metadata to isolate feature dimensions
    X = df.drop(columns=["label", "file_path", "class_name"], errors="ignore")
    # Target
    y = df["label"]
    return X, y

def main():
    try:
        X, y = load_data()
    except FileNotFoundError:
        print("Data file not found. Please ensure 'datasets/processed/features.csv' exists.")
        return
        
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression Model
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    # 2. KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
    knn_auc = auc(knn_fpr, knn_tpr)
    
    # Plotting ROC Curves
    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    # plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_auc:.3f})')
    
    # Diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.grid(alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()
