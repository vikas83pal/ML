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
# Simple Linear Regression with Visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Extract coefficients
slope = model.coef_[0]
intercept = model.intercept_

# Print model parameters
print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")

# Predict values for plotting the regression line
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x_range, y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Multiple Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Features: [Area (sq ft), Bedrooms, Age (years)]
X = np.array([
    [1500, 3, 20],
    [1700, 4, 15],
    [1300, 2, 30],
    [2000, 5, 10],
    [1600, 3, 25]
])

# Target: House Price (in $1000s)
y = np.array([300, 400, 250, 500, 330])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Print model details
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict for training data
y_pred = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='green', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.grid(True)
plt.tight_layout()
plt.show()


# 6. Decision Tree with Tuning
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

# Assuming X and y are defined
y_bin = (y > y.mean()).astype(int)

class_counts = np.bincount(y_bin)
cv_folds = min(5, class_counts.min())

params = {'max_depth': [3, 5, 7]}
cv = StratifiedKFold(n_splits=cv_folds)

clf = GridSearchCV(DecisionTreeClassifier(), params, cv=cv)
clf.fit(X, y_bin)

print("Best Decision Tree Params:", clf.best_params_)

# Train final model on full data with best params
best_depth = clf.best_params_['max_depth']
final_clf = DecisionTreeClassifier(max_depth=best_depth)
final_clf.fit(X, y_bin)

# Plot the decision tree
plt.figure(figsize=(12,8))
plot_tree(final_clf, filled=True, feature_names=[f"feature_{i}" for i in range(X.shape[1])], class_names=['0','1'])
plt.title("Decision Tree Visualization")
plt.show()


# 7. K-Nearest Neighbors
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Example dataset (replace with your data)
X, y = load_iris(return_X_y=True)

# If your dataset is very small, try fewer splits (like 2)
cv_splits = 3  # you can reduce this if you still get warnings

# StratifiedKFold preserves class distribution in each fold
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

# Find minimum class count to set max n_neighbors safely
unique, counts = np.unique(y, return_counts=True)
min_class_samples = counts.min()

print(f"Minimum samples in any class: {min_class_samples}")

# Define parameter grid ensuring n_neighbors <= min_class_samples
param_grid = {
    'n_neighbors': list(range(1, min(min_class_samples, 10) + 1))  # up to 10 or min_class_samples
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', error_score='raise')

grid_search.fit(X, y)

print("Best n_neighbors:", grid_search.best_params_['n_neighbors'])
print("Best CV Score:", grid_search.best_score_)


# 8. Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, (y > y.mean()).astype(int))
print("Logistic Regression Score:", log_reg.score(X, (y > y.mean()).astype(int)))

# 9. K-Means Clustering
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load example data
X, y = load_iris(return_X_y=True)

# Create binary target: 1 if y > mean, else 0
y_binary = (y > y.mean()).astype(int)

# Initialize Logistic Regression with higher max_iter for convergence
log_reg = LogisticRegression(max_iter=1000)

# Fit model
log_reg.fit(X, y_binary)

# Evaluate model accuracy on the same data
score = log_reg.score(X, y_binary)
print("Logistic Regression Score:", score)


# 10. Performance Analysis (Classification Report)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Create KMeans instance with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)

# Fit model to data
kmeans.fit(X)

# Predict cluster labels
y_kmeans = kmeans.predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering")
plt.show()
