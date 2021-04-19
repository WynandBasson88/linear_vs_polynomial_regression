# Linear Regression vs Polynomial Regression

# Relationship to model:
# x -> Explanitory variable
# y -> Response variable
# y = x^2 + x + 3
# For every person living in the Netherlands (x) there are (x^2 + x + 3) number of bicycles  

####################

# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model        # for LinearRegression()
from sklearn import preprocessing       # for PolynomialFeatures()

# Function for the best fit curve
def poly(dX_Train, dY_Train, x):
    return (x*x + m(np.squeeze(dX_Train), dY_Train)*x + b(dX_Train, dY_Train))

# Function for the best fit line
def y(dX_Train, dY_Train, x):
    return (m(np.squeeze(dX_Train), dY_Train)*x + b(dX_Train, dY_Train))

# Function for the gradient
def m(x_train, y_train):
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    product_mean = np.mean(x_train * y_train)
    x_mean_squared = math.pow(x_mean, 2)
    x_squared_mean = np.mean(x_train * x_train)
    return ((x_mean * y_mean) - product_mean)/(x_mean_squared - x_squared_mean)

# Function for the y-intercept
def b(x_train, y_train):
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    return (y_mean - (m(np.squeeze(x_train), y_train)*x_mean))

####################

# Start of the Algorithm

# Creating a figure with subplots
fig = plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

####################

# Data Lists
dx_train = [1, 3, 5, 7, 9, 11, 13]
dy_train = [5, 15, 33, 59, 93, 135, 185]
dx_test = [0, 2, 4, 6, 8, 10, 12, 14]
dy_test = [3, 9, 23, 45, 75, 113, 159, 213]

# Training data
print("TRAINING DATA:\n")

dX = np.array([dx_train])
print("dX:")
print(type(dX))
print(dX.shape)
print(dX)
print("")
dX_train = np.transpose(dX)
print("dX_train:")
print(type(dX_train))
print(dX_train.shape)
print(dX_train)
print("")

dY_train = np.array(dy_train)
print("dY_train:")
print(type(dY_train))
print(dY_train.shape)
print(dY_train)
print("")

# Test data
print("TEST DATA:\n")

dXX = np.array([dx_test])
print("dXX:")
print(type(dXX))
print(dXX.shape)
print(dXX)
print("")
dX_test = np.transpose(dXX)
print("dX_test:")
print(type(dX_test))
print(dX_test.shape)
print(dX_test)
print("")

dY_test = np.array(dy_test)
print("dY_test:")
print(type(dY_test))
print(dY_test.shape)
print(dY_test)
print("")

####################

# Linear Regression
lr = linear_model.LinearRegression()                                        # Create a linear regression object called lr
lr.fit(dX_train, dY_train)                                                  # Use dX_train & dY_train to train the algorithm

# Calculate the Mean square error
# Calculate the correlation between the explanatory and responsive variables
mse = np.mean((lr.predict(dX_test) - dY_test) **2)
lr_score = lr.score(dX_test, dY_test)

# Print the calculated values
print("Linear Regression Calculations:")
print("lr.coef: {}".format(lr.coef_))
print("mse: {}".format(mse))
print("lr_score: {}".format(lr_score))
print("")

####################

# Polynomial Regression
quadratic_featurizer = preprocessing.PolynomialFeatures(degree = 2)

dX_train_quad = quadratic_featurizer.fit_transform(dX_train)                # dX_train squared
print("dX_train squared:")
print(type(dX_train_quad))
print(dX_train_quad.shape)
print(dX_train_quad)
print("")

dX_test_quad = quadratic_featurizer.transform(dX_test)                      # dX_test squared
print("dX_test squared:")
print(type(dX_test_quad))
print(dX_test_quad.shape)
print(dX_test_quad)
print("")

pr = linear_model.LinearRegression()                                        # Create a linear regression object called pr
pr.fit(dX_train_quad, dY_train)                                             # Use dX_train_quad & dY_train to train the algorithm

# Calculate the Mean square error
# Calculate the correlation between the explanatory and responsive variables
mse_poly = np.mean((pr.predict(dX_test_quad) - dY_test) **2)
pr_score = pr.score(dX_test_quad, dY_test)

# Print the calculated values
print("Polynomial Regression Calculations:")
print("pr.coef: {}".format(pr.coef_))
print("mse_poly: {}".format(mse_poly))
print("pr_score: {}".format(pr_score))
print("")

####################

# Plot using inbuilt functions: lr.predict() & pr.predict()
ax1.set_title('Linear Regression using lr.predict()')
ax1.scatter(dX_train, dY_train, c ='r', label = 'Training data')
ax1.scatter(dX_test, dY_test, c ='g', label = 'Test data')
ax1.plot(dX_test, lr.predict(dX_test), c='b', label='Result')               # Use dX_test to predict the y value
ax1.legend(['Result', 'Training data', 'Test data'], loc=4)

ax2.set_title('Polynomial Regression using pr.predict()')
ax2.scatter(dX_train, dY_train, c ='r', label = 'Training data')
ax2.scatter(dX_test, dY_test, c ='g', label = 'Test data')
ax2.plot(dX_test, pr.predict(dX_test_quad), c='b', label='Result')          # Use dX_test_quad to predict the y value
ax2.legend(['Result', 'Training data', 'Test data'], loc=4)

# Plot using self defined functions: y() & poly()
ax3.set_title('Linear Regression using y = mx + b')
ax3.scatter(dX_train, dY_train, c='r', label='Training data')
ax3.scatter(dX_test, dY_test, c='g', label='Test data')
ax3.plot(dX_test, y(dX_train, dY_train, dX_test), c='b', label='Result')    # Use dX_train & dY_train to train the algorithm & dX_test to predict the y value
ax3.legend(['Result', 'Training data', 'Test data'], loc=4)

ax4.set_title('Polynomial Regression using y = x^2 + mx + b')
ax4.scatter(dX_train, dY_train, c='r', label='Training data')
ax4.scatter(dX_test, dY_test, c='g', label='Test data')
ax4.plot(dX_test, poly(dX_train*dX_train, dY_train, dX_test), c='b', label='Result')    # Use dX_train^2 & dY_train to train the algorithm & dX_test to predict the y value    
ax4.legend(['Result', 'Training data', 'Test data'], loc=4)

plt.show()
