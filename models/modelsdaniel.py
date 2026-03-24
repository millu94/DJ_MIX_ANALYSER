import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

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

def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train a Logistic Regression model and return its ROC curve metrics."""
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_auc = auc(lr_fpr, lr_tpr)
    print(f"Logistic Regression trained. AUC: {lr_auc:.3f}")
    return lr, lr_fpr, lr_tpr, lr_auc

def train_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=5):
    """Train a KNN model and return its ROC curve metrics."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
    knn_auc = auc(knn_fpr, knn_tpr)
    print(f"KNN (k={k}) trained. AUC: {knn_auc:.3f}")
    return knn, knn_fpr, knn_tpr, knn_auc

def plot_roc_curves(models_roc_data):
    """Plot ROC curves for multiple models. Expects a list of dictionaries."""
    plt.figure(figsize=(8, 6))
    for data in models_roc_data:
        plt.plot(data['fpr'], data['tpr'], label=f"{data['name']} (AUC = {data['auc']:.3f})")
    
    # Diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def graph_knn_overfitting(X_train_scaled, y_train, X_test_scaled, y_test, max_k=40):
    """Plot k vs Accuracy and Overfitting graphs for KNN."""
    k_values = range(1, max_k + 1)
    train_accuracies = []
    test_accuracies = []
    
    print(f"Computing KNN accuracy for k=1 to {max_k}...")
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)
        
        train_accuracies.append(knn_temp.score(X_train_scaled, y_train))
        test_accuracies.append(knn_temp.score(X_test_scaled, y_test))
        
    # Graph 1: k vs Accuracy (Testing)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, test_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Testing Accuracy')
    plt.title('KNN: k Value vs Testing Accuracy')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Graph 2: Overfitting check (Train vs Test Accuracy)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, train_accuracies, marker='o', linestyle='-', color='r', label='Training Accuracy')
    plt.plot(k_values, test_accuracies, marker='o', linestyle='-', color='b', label='Testing Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN: Overfitting Check (Train vs Test Accuracy)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

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
    
    # ==========================================
    # MODULAR EXECUTION: Uncomment what you want to run!
    # ==========================================
    
    # 1. Train Logistic Regression
    lr, lr_fpr, lr_tpr, lr_auc = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 2. Train KNN
    knn, knn_fpr, knn_tpr, knn_auc = train_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=5)
    
    # 3. Plot ROC Curves
    roc_data = [
        {'fpr': lr_fpr, 'tpr': lr_tpr, 'auc': lr_auc, 'name': 'Logistic Regression'},
        {'fpr': knn_fpr, 'tpr': knn_tpr, 'auc': knn_auc, 'name': 'KNN'}
    ]
    plot_roc_curves(roc_data)
    
    # 4. Graph KNN Accuracies and Overfitting
    graph_knn_overfitting(X_train_scaled, y_train, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
