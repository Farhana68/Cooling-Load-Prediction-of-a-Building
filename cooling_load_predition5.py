

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
# %matplotlib inline
#from google.colab import files
#uploaded = files.upload()
#import io
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

df = pd.read_csv('cooling_load1.csv')
print(df)

# print first 5 rows in the dataframe
df.head()

#print last 5 rows of the dataframe
df.tail()

# number of rows and colums
df.shape

# getting some basic infomation about the data
df.info()

# checking the number  missing values
df.isnull().sum()

# getting statisical measures of the data
df.describe()

correlation = df.corr()

# construting a heatmap to nderstand the correlation
plt.figure(figsize=(8,8))
#.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# correlation values of Cooling Load
print(correlation['Y1'])

# checking the distribution of the cooling load
#sns.distplot(df['Y1'],color='green')

X = df.drop(['Y1'],axis=1)
Y = df['Y1']

print(X)

print(Y)

"""Splitting into training and testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

"""Model Training

K Nearest Neighbors Regression
"""

from math import sqrt
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(metrics.mean_squared_error(Y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

# Loading The Model
knn_model = KNeighborsRegressor(2)

knn_model.fit(X_train,Y_train)

# accuracy for pediction on test data
test_data_prediction = knn_model.predict(X_test)

# R square error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 =  metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error :', score_2)

#Mean Squared Error
score_4 = metrics.mean_squared_error(Y_test, test_data_prediction)
print("Mean Squared Error :" ,score_4)
# Mean Absolute Percentage error
score_3 = (metrics.mean_absolute_percentage_error(Y_test, test_data_prediction)*100)
print("Mean Absolute Percentage Error :" ,score_3, "%")

"""Model Training

Linear Regression
"""

# Loading The Model
lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train,Y_train)

# accuracy for pediction on test data
test_data_prediction = lin_reg_model.predict(X_test)

# R square error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 =  metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error :', score_2)

#Mean Squared Error
score_4 = metrics.mean_squared_error(Y_test, test_data_prediction)
print("Mean Squared Error :" ,score_4)
# Mean Absolute Percentage error
score_3 = (metrics.mean_absolute_percentage_error(Y_test, test_data_prediction)*100)
print("Mean Absolute Percentage Error :" ,score_3, "%")

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Cooling Load")
plt.ylabel("Predicted Cooling Load")
plt.title("Actual Cooling Load vs Predicted Cooling Load")
#plt.show()

"""Model Training

Random Forest Regressor
"""

regressor = RandomForestRegressor(n_estimators=150)

# training the model
regressor.fit(X_train,Y_train)

# prediction on test data
test_data_pediction = regressor.predict(X_test)

print(test_data_pediction)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_pediction)
print("R squared error : ", error_score)

# Mean Absolute Error
score_mean_error =  metrics.mean_absolute_error(Y_test, test_data_pediction)
print('Mean Absolute Error :', score_mean_error)

#Mean Squared Error
score_4 = metrics.mean_squared_error(Y_test, test_data_pediction)
print("Mean Squared Error :" ,score_4)
# Mean Absolute Percentage error
score_3 = (metrics.mean_absolute_percentage_error(Y_test, test_data_pediction)*100)
print("Mean Absolute Percentage Error :" ,score_3, "%")

Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_pediction, color='green', label='Predicted Value')
plt.title('Actual Cooling Load vs Predicted cooling Load')
plt.xlabel('Number of values')
plt.ylabel('Cooling Load')
plt.legend()
#plt.show()

"""Model training

Support Vector Machine
"""

regressor = svm.SVR()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
svm_pred = regressor.predict(X_test)

print(svm_pred)

# R squared error
error_score = metrics.r2_score(Y_test, svm_pred)
print("R squared error : ", error_score)
# Mean Absolute Error
score_mean_error =  metrics.mean_absolute_error(Y_test, svm_pred)
print('Mean Absolute Error :', score_mean_error)

#Mean Squared Error
score_4 = metrics.mean_squared_error(Y_test, svm_pred)
print("Mean Squared Error :" ,score_4)
# Mean Absolute Percentage error
score_3 = (metrics.mean_absolute_percentage_error(Y_test, svm_pred)*100)
print("Mean Absolute Percentage Error :" ,score_3, "%")