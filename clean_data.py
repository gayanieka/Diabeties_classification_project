import pandas as pd

#Check for null values in every columns and remove them
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
print(df.isnull().sum())
df.dropna(inplace=True)

#remove duplicates
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

