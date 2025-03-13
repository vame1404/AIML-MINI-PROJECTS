import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:11]
y = BosData.iloc[:, 13] # MEDV: Median value of owner-occupied homes in $1000s








print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")

y_test_predict = reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)
print('Test RMSE =', rmse)
print('Test R2 score =', r2)
