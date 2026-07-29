import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'/Users/abhiramireddygali/Downloads/april(2025)/30th-1.SVR, KNN/Polynomial regression/1.POLYNOMIAL REGRESSION/emp_sal.csv')

X = dataset.iloc[:,1:2]
y = dataset.iloc[:,2]

from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Linear regression graph')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or Buff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

from sklearn.svm import SVR
svr_reg = SVR(kernel='poly',degree=4,gamma='auto')
svr_reg.fit(X,y)

svr_reg.predict([[6.5]])

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=4,weights='distance')
regressor_knn.fit(X, y)

regressor_knn.predict([[6.5]])

from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(criterion='absolute_error',splitter='best',random_state=1)
regressor_dtr.fit(X, y)

regressor_dtr.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(random_state=0)
regressor_rf.fit(X, y)

regressor_rf.predict([[6.5]])





























