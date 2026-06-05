import pandas as pd
import matplotlib.pyplot as plt

#Check for null values in every columns and remove them
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
print(df.isnull().sum())
df.dropna(inplace=True)

#remove duplicates
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

#Check for outliers using boxplot
cols_to_check = ['BMI', 'MentHlth', 'PhysHlth']
for col in cols_to_check:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# List of columns to clean
cols_to_clean = ['BMI', 'MentHlth', 'PhysHlth']

for col in cols_to_clean:
    # Calculate bounds
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # how many values fall outside these bounds
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}' has {len(outliers)} values outside the IQR bounds.")
    
    # Clip the values to the lower and upper bounds
    # This prevents extreme outliers from skewing your model without deleting the row
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("\nOutlier treatment complete.")