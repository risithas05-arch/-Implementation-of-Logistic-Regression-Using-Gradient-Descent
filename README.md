# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.1.Initialize slope m, intercept c, learning rate, and number of iterations.

2.Compute predicted output using y = mX + c.

3.Update m and c using gradient descent.

4.Repeat until convergence and display final parameters. *Regression line 



## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Vedha M 
RegisterNumber:25012201

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (X = input, y = output)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:risitha.s 
RegisterNumber:25018977  
*/
```

## Output
![WhatsApp Image 2026-02-13 at 14 16 06](https://github.com/user-attachments/assets/eee04731-78ee-479f-9a1f-7070f9e7a914)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

