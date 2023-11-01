# Import required dependencies
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Data Visualization
import pandas as pd # Manipulate DataFrames

# Load the Data - (Using Numpy)
data = np.loadtxt(r'machine_learning_course\exercise1_linear_regression\source_data\ex1data1.txt', delimiter = ',')
X = data[:, 0][:, np.newaxis] # Population
X = np.insert(X, 0, 1, axis = 1)
y = data[:, 1][:, np.newaxis] # Profit

print(X)
print(y)


# In this part of this exercise, you will implement linear regression with one variable to
#  predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are 
# considering different cities for opening a new outlet. The chain already has trucks in 
# various cities and you have data for profits and populations from the cities. You would like 
# to use this data to help you select which city to expand to next. The file ex1data1.txt 
# contains the dataset for our linear regression problem. The first column is the population 
# of a city and the second column is the profit of a food truck in that city. A negative value 
# for profit indicates a loss

# PLOTING THE DATA
def plotData(X, y):
    plt.figure(figsize = (11, 7))
    plt.plot(X[:, 1], y, 'rx', markersize = 10)
    plt.grid()
    plt.ylabel('Profit in $100,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

plotData(X, y)

# Load Data - (Using Pandas)
data = pd.read_csv(r'machine_learning_course\exercise1_linear_regression\source_data\ex1data1.txt')
print(data.shape)
print(data.head())

print(data.columns)

# Add Column Headers
data.columns = ['0', '1']

X = data[['0']].reset_index(drop=True)
y = data[['1']].reset_index(drop=True)

# Convert to Numpy Array
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
print(X)
print(y)


# PLOTING THE DATA - IN THE PANDAS CASE
def plotData(X, y):
    plt.figure(figsize=(11, 7))
    plt.plot(X, y, 'rx', markersize=10)
    plt.grid()
    plt.ylabel('Profit in $100,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

plotData(X, y)

# 5x5 Identity Matrix
def warmupExercise():
    return np.identity(5)

warmupExercise()

# The objective of linear regression is to minimize the cost function:
# J(\Theta) = \frac{1}{2m} \sum_{i=1}^m (h_{\Theta}(x^{(i)}) - y^{(i)})^2

# Where the hypothesis function is given by the linear model:
# h_{\Theta}(x) = \vec{\Theta}^{\top} \vec{x}
def computeCost(X, y, theta):
    m = y.size
    predictions = X.dot(theta)
    sqerror = (predictions - y)**2
    J = 1/(2*m) * np.sum(sqerror)
    return J

theta1 = np.zeros((2,1))
theta2 = np.array([[-1],[2]])
a = computeCost(X, y, theta1)
b = computeCost(X, y, theta2)
print('The Predicted Parameters are:' + '\n' + '%0.2f and %0.2f'%(a,b))

# In batch gradient descent, each iteration performs the update

# repeat until convergence: {
#   theta_j := theta_j - alpha * 1/m * sum(h_theta(x^{(i)}) - y^{(i)}) * x^{(i)}_j
# }
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = []
    for _ in range(num_iters):
        errors = X.dot(theta) - y
        delta = np.dot(X.T, errors)
        theta = theta - ((alpha / m) * delta)
        J_history.append(computeCost(X, y, theta))
        return theta, J_history