import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:]
y = dum_df.iloc[:,0]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
