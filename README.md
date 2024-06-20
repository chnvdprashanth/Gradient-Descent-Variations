# Gradient Descent Variations for Linear Regression
## Description
This repository contains implementations of three different gradient descent variations for both linear regression and logistic regression models. The variations include:
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
## Installation
To use the code in this repository, you need to have Python installed on your system. Additionally, you can install the required packages using pip.

```bash
  pip install -r requirements.txt
```

## Usage
Each gradient descent variation is implemented as a separate Python script. You can run these scripts to see how each algorithm performs on a sample dataset.

## Running the code

```bash
  python batch_gradient_descent.py
```
Similarly, you can run the other variations:

```bash
  python stochastic_gradient_descent.py
  python mini_batch_gradient_descent.py
```

## Examples

## Linear Regression 

```python

from batch_gradient_descent import Gradient-Descent-Variations
import numpy as np

# Sample data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Model initialization and training using Batch Gradient Descent
bgd = mutliVarGrad(learning_rate=0.01, n_iterations=1000)
bgd.fit(X, y)

# Using Stochastic Gradient Descent
sgd = SGD_Regression(learning_rate=0.01, n_iterations=1000)
sgd.fit(X,y)

# Using Mini-Batch Gradient Descent
mbgd = MBGD_Regressor(learning_rate=0.01, n_iterations=1000)
mgd.fit(X,y)

# Predictions
predictions_bgd = bgd.predict(X)
predictions_sgd = sgd.predict(X)
predictions_mbgd = mbgd.predict(X)

print(predictions_bgd)
print(predictions_sgd)
print(predictions_mbgd)

```
