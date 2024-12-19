import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Analyze Data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedianHouseValue"] = data.target

print("Dataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# 2. Handle Missing Values
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# 3. Exploratory Data Analysis (EDA)
sns.set_style("whitegrid")
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Feature Distributions")
plt.show()

# Scatter plots for key features
features = ["MedInc", "AveRooms", "AveOccup"]
for feature in features:
    sns.scatterplot(x=df[feature], y=df["MedianHouseValue"])
    plt.title(f"{feature} vs MedianHouseValue")
    plt.show()

# 4. Feature Engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["MedInc", "HouseAge"]])
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(["MedInc", "HouseAge"]))
df = df.drop(["MedInc", "HouseAge"], axis=1).join(df_poly)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop("MedianHouseValue", axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.drop("MedianHouseValue", axis=1).columns)
df_scaled["MedianHouseValue"] = df["MedianHouseValue"]

# 5. Split Data
X = df_scaled.drop("MedianHouseValue", axis=1)
y = df_scaled["MedianHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training with Hyperparameter Tuning
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
}

params = {
    "Ridge": {"alpha": [0.1, 1, 10]},
    "Lasso": {"alpha": [0.1, 1, 10]},
    "RandomForestRegressor": {"n_estimators": [50, 100], "max_features": ["auto", "sqrt"]},
}

for model_name, model in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("regressor", model),
    ])

    grid = GridSearchCV(pipeline, params.get(model_name, {}), cv=5, scoring="neg_mean_squared_error", error_score='raise')
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} MSE: {mse}, Best Params: {grid.best_params_}")

# 7. Custom Model Implementation
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.random.randn(X_b.shape[1], 1)
        for _ in range(self.iterations):
            gradient = 2 / X_b.shape[0] * X_b.T.dot(X_b.dot(self.weights) - y.reshape(-1, 1))
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
custom_model = CustomLinearRegression(learning_rate=0.01, iterations=1000)
custom_model.fit(X_train_np, y_train_np)
y_pred_custom = custom_model.predict(X_test_np)
custom_mse = mean_squared_error(y_test_np, y_pred_custom)
print(f"Custom Model MSE: {custom_mse}")
