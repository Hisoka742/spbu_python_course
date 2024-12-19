from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedianHouseValue"] = data.target

# Handle missing values by imputing the mean
df.iloc[:, :-1] = SimpleImputer(strategy="mean").fit_transform(df.iloc[:, :-1])

# Features and target
X = df.drop("MedianHouseValue", axis=1)
y = df["MedianHouseValue"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
}

# Define parameter grids with proper prefixes
param_grids = {
    "LinearRegression": {},  # No hyperparameters for Linear Regression
    "Ridge": {"regressor__alpha": [0.1, 1, 10]},
    "Lasso": {"regressor__alpha": [0.1, 1, 10]},
    "RandomForestRegressor": {
        "regressor__n_estimators": [50, 100],
        "regressor__max_features": ["auto", "sqrt"],
    },
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Define the pipeline
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
            ("scaler", StandardScaler()),  # Standardize features
            ("regressor", model),  # Model
        ]
    )

    # Perform GridSearchCV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids.get(
            model_name, {}
        ),  # Get the parameter grid for the model
        scoring="neg_mean_squared_error",
        cv=5,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = grid.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"{model_name} - Best Params: {grid.best_params_}, MSE: {mse}")
