Diabetes Risk Prediction Model
1. Overview
This project leverages machine learning to predict diabetes risk levels using health survey data. By analyzing key demographic and physiological markers, the model provides a classification of patients into "No Diabetes," "Pre-diabetic," or "Diabetic" categories.

2. Project Goals
Identify high-risk individuals for early clinical screening.

Understand the relationship between lifestyle/health factors (like BMI and High Blood Pressure) and diabetes progression.

Develop a robust predictive model that prioritizes recall to ensure high-risk cases are not overlooked.

3. Data Insights
Our Exploratory Data Analysis (EDA) revealed significant correlations:

Physical Health: A clear, statistically significant increase in "poor physical health days" as patients move from non-diabetic to diabetic status.

Key Predictors: Feature importance analysis identified BMI, Age, and General Health as the primary drivers of diabetes risk.

4. Methodology
Model: Random Forest Classifier (chosen for its handling of non-linear health data).

Handling Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset and prevent bias toward the majority class.

Evaluation: Optimized for Recall to minimize false negatives in medical diagnostics.

5. Technical Requirements
To run this project, you will need Python 3.x and the following libraries:

Bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn shap
6. How to Run
Clone this repository.

Install the required dependencies.

Run the main_analysis.ipynb (or your script name) to generate the model and visualizations.

7. Results
The model successfully identifies diabetes patterns using a multi-class classification approach.
