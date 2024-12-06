import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Step 1: Load and Prepare the Data
iris = load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# Step 2: Exploratory Data Analysis (EDA)
sns.pairplot(iris_df, hue="target")
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Step 3: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Feature Engineering
# Adding interaction features
iris_df["petal area"] = iris_df["petal length (cm)"] * iris_df["petal width (cm)"]
iris_df["sepal area"] = iris_df["sepal length (cm)"] * iris_df["sepal width (cm)"]
X_engineered = scaler.fit_transform(iris_df.drop("target", axis=1))

# Step 5: Model Implementation and Hyperparameter Tuning
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "k-NN": KNeighborsClassifier(),
}
params = {
    "Logistic Regression": {"C": [0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]},
    "SVM": {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"]},
    "Decision Tree": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
    "k-NN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
}
best_estimators = {}

for name, model in models.items():
    clf = GridSearchCV(model, params[name], cv=5, scoring="accuracy")
    clf.fit(X_engineered, y)
    best_estimators[name] = clf.best_estimator_
    print(f"{name}: Best Params: {clf.best_params_}, Best Score: {clf.best_score_}")

# Step 6: Model Evaluation
for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_engineered)
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {name}:\n", cm)
    print(f"Classification Report for {name}:\n", classification_report(y, y_pred))

# Step 7: Implement a simple k-NN classifier
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Hyperparameter tuning for the custom k-NN
knn = SimpleKNN(k=5)
knn.fit(X_scaled, y)
y_pred = knn.predict(X_scaled)
print("Custom k-NN Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Custom k-NN Classification Report:\n", classification_report(y, y_pred))
