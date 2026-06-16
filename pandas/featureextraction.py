import numpy as np
import pandas as pd
df = pd.read_csv("anime.csv")
#print(df.describe)
#print(df)

#print(df.loc[1])

def extractfeature(txt):
    data=""
    check=False
    for i in txt:
        if i ==")":
           break
        if i == '(':
            check=False
            return data
        if check==True:
            data = data+i
    return data           
df["Episode"] = df["Title"].apply(extractfeature)
print(df)

