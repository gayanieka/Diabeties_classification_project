from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from predictive_modeling import X_test, X_train
from predictive_modeling import y_test, y_train

# 1. Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 2. Re-train the model on the balanced data
model_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
model_balanced.fit(X_resampled, y_resampled)

# 3. Evaluate again
y_pred_balanced = model_balanced.predict(X_test)
print(classification_report(y_test, y_pred_balanced))


