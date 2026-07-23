import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ab_nyc_2019.csv")

print(df.head())

df.tail()

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.info())

print(df.describe())
print(df.describe(include='object'))

#print num cols
num_cols = df.select_dtypes(include=['int64','float64']).columns
print(num_cols)

cat_cols = df.select_dtypes(include=['object']).columns
print(cat_cols)

#check the null values
print(df.isnull().sum())

#check the duplicate rows
print(df.duplicated().sum())

#check the unique values
print(df.nunique())

#Exploring Categorical values
print(df['room_type'].unique())

print(df['room_type'].value_counts())

print(df['neighbourhood_group'].value_counts())

#identify the target variable
y= df['price']

#identify input features
X = df.drop(columns=['price'])


#Data Cleaning
#calculate the missing value percentage
missing_precentage = (df.isnull().sum()/len(df))*100
missing_precentage.sort_values(ascending=False)
print(missing_precentage)

#visulize missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing values heat map")
plt.show()

#handling missing values
df['name'].fillna('Unknown')

df['host_name'].fillna('Unknown')


df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].mean())



print(df.isnull().sum())
print("duplicate value")
print(df.duplicated().sum())
duplicates = df[df.duplicated()]

print(duplicates)
#drop duplicated
df.drop_duplicates(inplace=True)

print(df.duplicated().sum())

#check the chapter 3
print(df.dtypes)


df['last_review'] = pd.to_datetime(df['last_review'])
df.info()

df.drop(columns=['id','host_id','name','host_name'],inplace=True)

print(df.columns)

print(df.shape)


cdf = df

print(cdf.head())



#EDA 
# Display all numerical columns

print(cdf.select_dtypes(include=['int64', 'float64']).columns)

print(cdf.select_dtypes(include=['object']).columns)

#price distribution
plt.figure(figsize=(10,6))
sns.histplot(cdf['price'],bins=50,kde=True)
plt.title("Price distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#box plot for outlier
plt.figure(figsize=(10,5))
sns.boxplot(x=cdf['price'])
plt.title("Box plot price")
plt.show()

#distrubtion of Minimum nights
plt.figure(figsize=(10,2))
sns.histplot(cdf['minimum_nights'],bins=50,kde=True)
plt.title("Minimun Nigths Distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=cdf['minimum_nights'])
plt.show()

#review per month hisplot
plt.figure(figsize=(10,6))
sns.histplot(cdf['reviews_per_month'],bins=30,kde=True)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(cdf['availability_365'],bins=30)
plt.show()

#categorical analysis
#Room type Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='room_type',data=cdf)
plt.title("Room Type Distribution")
plt.show()

#Pie chart
room_counts = cdf['room_type'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(room_counts,labels=room_counts.index,autopct='%1.1f%%')
plt.title("Room Type Percentage")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='neighbourhood_group',data=cdf)
plt.show()