# 1. Central Tendency & Dispersion
import statistics as stats

data = [10, 20, 20, 30, 40, 50, 60]
print("Mean:", stats.mean(data))
print("Median:", stats.median(data))
print("Mode:", stats.mode(data))
print("Variance:", stats.variance(data))
print("Standard Deviation:", stats.stdev(data))

# 2. Python Basic Libraries
import math
import numpy as np
from scipy import stats as scipy_stats

print("Math sqrt(16):", math.sqrt(16))
print("Numpy array mean:", np.mean(data))
print("Scipy mode:", scipy_stats.mode(data, keepdims=True).mode[0])

# 3. Pandas & Matplotlib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'values': data})
df.plot(kind='bar')
plt.title("Bar Plot of Data")
plt.show()

# 4. Simple Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression().fit(x, y)
print("Simple Linear Regression Coef:", model.coef_)
print("Intercept:", model.intercept_)

# 5. Multiple Linear Regression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X, y = boston.data, boston.target
model = LinearRegression().fit(X, y)
print("Multiple Regression Score:", model.score(X, y))

# 6. Decision Tree with Tuning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {'max_depth': [3, 5, 7]}
clf = GridSearchCV(DecisionTreeClassifier(), params)
clf.fit(X, (y > y.mean()).astype(int))
print("Best Decision Tree Params:", clf.best_params_)

# 7. K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, (y > y.mean()).astype(int))
print("KNN Score:", knn.score(X, (y > y.mean()).astype(int)))

# 8. Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, (y > y.mean()).astype(int))
print("Logistic Regression Score:", log_reg.score(X, (y > y.mean()).astype(int)))

# 9. K-Means Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)
print("KMeans Cluster Centers:", kmeans.cluster_centers_)

# 10. Performance Analysis (Classification Report)
from sklearn.metrics import classification_report

y_pred = log_reg.predict(X)
print("Classification Report:\n", classification_report((y > y.mean()).astype(int), y_pred))