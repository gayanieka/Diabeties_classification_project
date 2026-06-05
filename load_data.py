import pandas as pd

df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
print(df.shape)
print(df.dtypes)