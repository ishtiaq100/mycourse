import numpy as np
import pandas as pd
#using dictioner
data = {"Name":['Allen','Smit','Waliam'],
        "Age":[20,30,40],
        "City":['New york','New Joursey','Texila'],
        "Salary":[15000,20000,30000]
}
print(pd.DataFrame(data))

#using list
listdata=[
    ['johon',20,'New York',25000],
    ['smith',30,'centiygo',35000],
    ['wallen',40,'New joursey',2000]
]
#pd.DataFrame(listdata)
#add the column
columns = ["Name","Age","City","Salaries"]
#adding the custom own columns
df2= pd.DataFrame(listdata,columns=columns)
print(df2)
#print(df2[['Name','City']])
#df2['Desg']=['Manager','SalesMan','Developer']
#print(df2)
#removed the designation
#df2.drop(["Desg"],axis=1,inplace=True)
print(df2)

#select column
#print(df2['Name'])
#print(df2[['Name','City']])
#selection row
print(df2.loc[1])
print(df2.iloc[1])
print(df2.loc[[0,1]][['Name','City']])
print(df2[df2['Age']<30])