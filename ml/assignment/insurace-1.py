# import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv("insurance.csv")
print(df)
df.info()
#change data type into flocate
#encoding the column and change type
df['sex'] = df['sex'].map({
    'male':1,
    'female':0
}).astype(float)
#check the column is null values
print(df.isnull().sum())
print(df.columns)
df.info()

df['region']= df['region'].map({
    'southwest':1, 'southeast':2, 'northwest':3,'northeast':4
}).astype(float)

df['smoker']= df['smoker'].map({
    'yes':1, 'no':0
}).astype(float)
print(df['region'].unique())


#scalling the column of bmi
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df['bmi'] = scaler.fit_transform(df[['bmi']])



corr_with_target = df.corr(numeric_only=True)['charges'].sort_values(ascending=False)

print(corr_with_target)

input("Waiting here")


#calculate corlation matrix
plt.figure(figsize=(4,6))

sns.heatmap(df.corr(numeric_only=True)[['charges']],
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Features vs Charges")

plt.show()


#dive df into x,y
x= df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y=df['charges']

#create the regration plot
variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'] 
for var in variables:
    sns.regplot(data=df,x=var,y='charges')
    plt.show()

print(df.head())    


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
meanabsoluteerror = metrics.mean_absolute_error(y_test,predection)
meansqureerror = metrics.mean_squared_error(y_test,predection)
print(f"R square error : {r2:.4f}")
print(f"Mean Absoulte error : {meanabsoluteerror:.4f}")
print(f"Root mean squere error :{meansqureerror:.4f}")