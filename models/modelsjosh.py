import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import os

import joblib

def load_data(filepath="datasets/processed/djmix_dataset_partition_features.csv"):
    if not os.path.exists(filepath):
        filepath = "../datasets/processed/djmix_dataset_partition_features.csv"
    df = pd.read_csv(filepath)
    return df

def main():
    df = load_data()
    
    # 1. Create the 'mix_group' column to identify which DJ is which
    # This matches the logic from your pipeline.py
    df['mix_group'] = df['file_path'].apply(lambda x: os.path.basename(x).split('_')[1])

    # 2. Define the cohorts
    test_mixes = ['RA.1002Nooriyah', 'RA.989Binh']
    val_mixes  = ['RA.1030MainPhase']
    
    # 3. Create the DataFrames
    test_df  = df[df['mix_group'].isin(test_mixes)]
    val_df   = df[df['mix_group'].isin(val_mixes)]
    train_df = df[~df['mix_group'].isin(test_mixes + val_mixes)]

    print(f"--- Manual Split Summary ---")
    print(f"Training on: {train_df['mix_group'].unique()}")
    print(f"Testing on:  {test_df['mix_group'].unique()} (Nooriyah & Binh)")

    # 4. Separate Features (X) and Labels (y)
    # We drop metadata columns so the model only sees the audio numbers
    drop_cols = ["label", "file_path", "class_name", "mix_group"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["label"]
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["label"]

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- MODEL TRAINING ---
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    # 2. KNN
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
    knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
    knn_auc = auc(knn_fpr, knn_tpr)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, lw=2, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot(knn_fpr, knn_tpr, lw=2, label=f'KNN (AUC = {knn_auc:.3f})')
    plt.plot(rf_fpr, rf_tpr, lw=2, label=f'Random Forest (AUC = {rf_auc:.3f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Manual Split (Test on Unseen DJs)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    # Save the professional model and scaler for the user app
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(rf, os.path.join(model_dir, "production_rf_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "production_scaler.joblib"))
    print("\n💾 Model and Scaler saved to /models for production use!")

if __name__ == "__main__":
    main()
    