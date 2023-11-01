# Import required dependencies
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Data Visualization


# Load the Data
data = np.loadtxt(r'machine_learning_course\exercise1_linear_regression\source_data\ex1data1.txt', delimiter = ',')
X = data[:, 0][:, np.newaxis] # Population
X = np.insert(X, 0, 1, axis = 1)
y = data[:, 1][:, np.newaxis] # Profit

print(X)
print(y)
