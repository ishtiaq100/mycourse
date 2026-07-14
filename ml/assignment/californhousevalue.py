#https://www.kaggle.com/datasets/camnugent/california-housing-prices
# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv("housing.csv")
print(df)
df.info()

print(df.isnull().sum())

totalbedromsmean = df['total_bedrooms'].mean()
print(totalbedromsmean)
df['total_bedrooms']= df['total_bedrooms'].fillna(totalbedromsmean)
print(df.isnull().sum())
print(df['ocean_proximity'].unique())
#encoding the column
df['ocean_proximity']= df['ocean_proximity'].map({
 'NEAR BAY':1,'<1H OCEAN':2,'INLAND':3,'NEAR OCEAN':4,'ISLAND':5   
}).astype(float)
print(df.info())
print(df.tail(25))

#standard scaller
from sklearn.preprocessing import StandardScaler
numaricalcols = ['longitude','latitude','housing_median_age','total_rooms','median_house_value','total_bedrooms','population','households','median_income','ocean_proximity']
scaler = StandardScaler()

df[numaricalcols] = scaler.fit_transform(df[numaricalcols])
print(df.head())

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True)[['median_house_value']], annot=True,
            cmap='coolwarm',
            fmt='.2f')
plt.title("Features vs medain_house_value")
plt.show()


#making regration plot
variables = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'] 
for var in variables:
    sns.regplot(data=df,x=var,y='median_house_value')
    plt.show()

x = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'] ]
y= df['median_house_value']
#machine learning code start here
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#now data spilit with train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lin_reg.fit(x_train,y_train)

#now model predection
predection = lin_reg.predict(x_test)

#compare the result
compareresult = pd.DataFrame({
    'Actual':y_test,"Prediction":predection
})
print(compareresult.head())



#check the model accurracy 
from sklearn import metrics
r2 = metrics.r2_score(y_test,predection)
mae = metrics.mean_absolute_error(y_test,predection)
rmse = np.sqrt(metrics.mean_squared_error(y_test,predection))
lin_accuracy = metrics.r2_score(y_test,predection)*100
print(f"R Square score : {r2:.4f}") #eual to 1 is best
print(f"Mean absolute error : {mae:.4f}") #jitna 0 kaiy pass utna acha model
print(f"Root Mean Square error : {rmse:.4f}") #jitna 0 kaiy pass utna acha////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
print(f"Linnear Regresstion Accuracy:{lin_accuracy:.2f}%")