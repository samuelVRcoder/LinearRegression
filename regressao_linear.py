import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression

x, y = make_regression(n_samples = 100, n_features=1, noise=30)

plt.scatter(x, y)

modelo = LinearRegression()

modelo.fit(x, y)

a_coef = modelo.coef_

l_coef = modelo.intercept_

plt.plot(x, l_coef + a_coef * x, color = 'green')

plt.scatter(1.5, l_coef + a_coef * 1.5, color = 'yellow')

plt.show()
