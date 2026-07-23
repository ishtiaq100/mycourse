import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#fetch the default dataset boston
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston',version=1,as_frame=True,parser='pandas')

df = boston.frame
#set the target
df['price']= boston.target
print(df.head())

#dividing the dataset into x and y
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.head())
print(y.head())


#Linnear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

x = x.astype(float)
y = y.astype(float)

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg,x,y,scoring ='neg_mean_squared_error',cv=5)
print(mse)

mean_mse = mse.mean()

print("MSE scores per fold: ",mse)
print("Avg Negtive menan Sequare",mean_mse)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lin_reg.fit(x_train,y_train)

#predection
predection = lin_reg.predict(x_test)

comp_df = pd.DataFrame({'Actual':y_test,'Predicted':predection})
print(comp_df.head())


#matrics
from sklearn import metrics
r2 = metrics.r2_score(y_test,predection) #best score 1.0

mae = metrics.mean_absolute_error(y_test,predection)

#standart deviation of predeictive error
rmse = np.sqrt(metrics.mean_squared_error(y_test,predection))

print(f"R seaquare score : {r2: .4f}")

print(f"Mean Absoute Error(MAE) : {mae: .4f}")

print(f"Root Mean Square error(RMSE) : {rmse: .4f}")


#Ridge Regression Algo

