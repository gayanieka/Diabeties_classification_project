import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("cleaned_diabetes_data.csv")
df.info()


#Compare BMI across the three classes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', data=df)
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#Check how Smoker status relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', hue='Smoker', data=df)
plt.title('BMI Distribution by Diabetes Status and Smoker Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.ylabel('BMI')
plt.show()

#Check how physical activity status relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', hue='PhysActivity', data=df)
plt.title('BMI Distribution by Diabetes Status and Physical Activity')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.ylabel('BMI')
plt.legend(title='PhysActivity', labels=['Inactive', 'Active'])
plt.show()



#Check how age relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='Age', data=df)
plt.title('Age Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#One way anova for age and diabetes status
# Separate the age data into three groups
age_0 = df[df['Diabetes_012'] == 0]['Age']
age_1 = df[df['Diabetes_012'] == 1]['Age']
age_2 = df[df['Diabetes_012'] == 2]['Age']
# Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(age_0, age_1, age_2)
print(f"ANOVA Results for Age:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")
# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis. There is a statistically significant difference in age between groups.")
else:
    print("Result: Fail to reject the null hypothesis. Age differences are not statistically significant.")





#check how high blood pressure relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='BMI', hue='HighBP', data=df)
plt.title('BMI Distribution by Diabetes Status and High Blood Pressure')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.ylabel('BMI')
plt.legend(title='HighBP', labels=['No', 'Yes'])
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

#check how NoDocbcCost relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='NoDocbcCost', data=df)
plt.title('Diabetes Status by No Doctor Cost')
plt.show()

#check how diff walking relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='DiffWalk', data=df)
plt.title('Diabetes Status by Difficulty Walking')
plt.show()

#check how sex relates to Diabetes
plt.figure(figsize=(8, 5))
sns.countplot(x='Diabetes_012', hue='Sex', data=df)
plt.title('Diabetes Status by Sex')
plt.show()

#check how education relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='Education', data=df)
plt.title('Education Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()

#check how income relates to Diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_012', y='Income', data=df)
plt.title('Income Distribution by Diabetes Status')
plt.xlabel('Diabetes Status (0=None, 1=Pre, 2=Diabetes)')
plt.show()


#heatmap of correlation to understand bettween in data in the dataset
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



