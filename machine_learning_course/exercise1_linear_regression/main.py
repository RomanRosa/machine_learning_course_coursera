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