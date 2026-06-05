import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("cleaned_diabetes_data.csv")

# 1. Prepare Data
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# 2. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
