import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/home/gaurav/AI/Proiba_ML/Classification/Classification Data/insurance.csv")

print(df.head())

X = df[['age']]
Y = df[['insurance status']]

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.1)

model = LogisticRegression()

model.fit(train_x,train_y)


pred = model.predict(test_x)

print(pred)

print(test_y)