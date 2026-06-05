import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("cleaned_diabetes_data.csv")

df.info()


#Compare BMI across the three classes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', data=df)
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#Check how Smoker status relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='Smoker', data=df)
plt.title('Diabetes Status by Smoker')
plt.show()

#Check how physical activity status relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='PhysActivity', data=df)
plt.title('Diabetes Status by Physical Activity')
plt.show()

#Check how age relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='Age', data=df)
plt.title('Age Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how high blood pressure relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='HighBP', data=df)
plt.title('Diabetes Status by High Blood Pressure')
plt.show()

#check how high cholesterol relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='HighChol', data=df)
plt.title('Diabetes Status by High Cholesterol')
plt.show()

#check how general health relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='GenHlth', data=df)
plt.title('General Health Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how physical health relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='PhysHlth', data=df)
plt.title('Physical Health Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how mental health relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='MentHlth', data=df)
plt.title('Mental Health Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how HvyAlcoholConsump relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='HvyAlcoholConsump', data=df)
plt.title('Sleep HvyAlcoholConsump Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how AnyHealthcare relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='AnyHealthcare', data=df)
plt.title('Diabetes Status by AnyHealthcare')
plt.show()


#heatmap of correlation to understand bettween in data in the dataset
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


