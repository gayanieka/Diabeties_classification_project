from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from balance_data import X_resampled

# Define a grid of parameters to test
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Run the grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='recall_macro')
grid_search.fit(X_resampled, y_resampled)

print(f"Best Parameters: {grid_search.best_params_}")