import sklearn.datasets as dt

from sklearn.linear_model import LinearRegression as lr

from sklearn.model_selection import train_test_split as tts

model = lr()

X, y = dt.make_regression(n_samples=30000)

xtr, xts, ytr, yts = tts(X, y, random_state=42)

model.fit(xtr, ytr)

print(model.score(xts, yts))
