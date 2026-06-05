
import shap 

# Calculate SHAP values for the model
explainer = shap.TreeExplainer(model_balanced) 
shap_values = explainer.shap_values(X_test)

# Plot the summary
shap.summary_plot(shap_values, X_test)