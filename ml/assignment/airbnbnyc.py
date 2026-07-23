#https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv
#import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ab_nyc_2019.csv")

print(df.head())

print(df.tail())

print(df.shape)

print(df.info())

print(df.describe())
print(df.describe(include='object'))

#check the value of null

print(df.isnull().sum())

print((df.isnull().sum()/len(df))*100)

#check the duplicate values
print(df.duplicated().sum())

#check the uniques values
print(df.nunique())

#Data Cleaning
df['name'] = df['name'].fillna("Unknown")

df['host_name'] =df['host_name'].fillna("UnKnown")


df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())

df['last_review'] = pd.to_datetime(df['last_review'])

df['last_review'] = df['last_review'].fillna(df['last_review'].mean()) 
print(df.isnull().sum())

#outlier detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['price'])
plt.show()

sns.boxplot(x=df['minimum_nights'])
plt.show()


#distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'],bins=50)
plt.show()

sns.histplot(df['availability_365'],bins=30)
plt.show()

#step10 corrlation map
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.show()


#step11 category analysis
sns.countplot(data=df,x='room_type')
plt.show()

sns.countplot(data=df,x='neighbourhood_group')
plt.xticks(rotation=45)
plt.show()

#step 12 target analysis
print(df.groupby('room_type')['price'].mean())

sns.barplot(data=df,x='room_type',y='price')
plt.show()

print(df.groupby('neighbourhood_group')['price'].mean())

#step 13 scatter plt
plt.figure(figsize=(10,8))
sns.scatterplot(data=df,x='longitude',y='latitude',hue='room_type')
plt.show()


#feature eng
df['review_year'] = df['last_review'].dt.year
df['review_month'] = df['last_review'].dt.month
df['review_day'] = df['last_review'].dt.day
df['review_dayofweek'] = df['last_review'].dt.day_name()
df['is_weekend']=df['last_review'].dt.dayofweek>=5
df['is_weekend']=df['is_weekend'].astype(int)

df['price_category']=pd.qcut(
    df['price'],
    q=4,
    labels=['Low','Medium','High','Luxury']
)
df['availability_category']=pd.cut(
    df['availability_365'],
    bins=[-1,90,180,270,365],
    labels=['Low','Medium','High','Very High']
)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['room_type']=le.fit_transform(df['room_type'])
df['neighbourhood_group']=le.fit_transform(df['neighbourhood_group'])
df['neighbourhood']=le.fit_transform(df['neighbourhood'])


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

cols=[
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'availability_365'
]

df[cols]=scaler.fit_transform(df[cols])

df.drop(
    columns=[
        'id',
        'host_id',
        'host_name',
        'name',
        'last_review'
    ],
    inplace=True
)

print(df.head())