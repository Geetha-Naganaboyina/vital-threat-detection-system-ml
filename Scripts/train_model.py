import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import os

# Create directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_model():
    if not os.path.exists('data/health_data.csv'):
        print("Error: data/health_data.csv not found. Run generate_data.py first.")
        return

    print("Loading data and training model...")
    df = pd.read_csv('data/health_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest with specific parameters
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    
    if round(acc, 2) == 0.97:
        print("Model reached desired accuracy.")
    else:
        print(f"Warning: Accuracy is {acc:.4f}, expected ~0.97")

    # Save model and metrics
    joblib.dump(model, 'models/health_model.pkl')
    
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'recall': recall,
        'total_samples': len(df),
        'safe_samples': len(df[df['target'] == 0]),
        'not_safe_samples': len(df[df['target'] == 1])
    }
    joblib.dump(metrics, 'models/metrics.pkl')
    print("Model and metrics saved.")

if __name__ == "__main__":
    train_model()
