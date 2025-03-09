import requests
import json
import pandas as pd
from pandas import DataFrame
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection,  svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

response=requests.get("https://api.covid19india.org/data.json")
j=response.json()

df=pd.DataFrame(j["cases_time_series"])
df=df[["totalconfirmed","totaldeceased"]]
df['date'] = pd.date_range(start='30/1/2020', periods=len(df), freq='D')
df.set_index("date", inplace=True)
print(df)

forecast_col="totalconfirmed"
#forecast_col1="totaldeceased"
df.fillna(-99999, inplace=True)
forecast_out=int(math.ceil(0.021*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)
#df['label1']=df[forecast_col1].shift(-forecast_out)
#print(df)

X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

df.dropna(inplace=True)
y=np.array(df['label'])
#X1=np.array(df.drop(['label1'],1))
#y1=np.array(df['label1'])
#X1=preprocessing.scale(X1)
#y1=np.array(df['label1'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(X1, y1, test_size=0.2)

clf= LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy=clf.score(X_test, y_test)
forecast_set=clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast']=np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df=df.astype(float)
df['totalconfirmed'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('No. of Cases')
plt.show()
