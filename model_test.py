import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Hotel_Reservations.csv')
df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

X = df.drop(columns=['avg_price_per_room','Booking_ID'])
y = df['avg_price_per_room'].values

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])
    ],
    remainder='passthrough'
)
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_categorical = ct.named_transformers_['encoder'].categories_
num_categorical_columns = sum(len(cats) for cats in num_categorical)

sc = StandardScaler()
X_train[:, num_categorical_columns:] = sc.fit_transform(X_train[:, num_categorical_columns:])
X_test[:, num_categorical_columns:] = sc.transform(X_test[:, num_categorical_columns:])

poly_degree = 2

models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(kernel='rbf'),
    "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(n_estimators=10, random_state=42)
}

results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "RMSE": rmse, "R2 Score": r2}
    except Exception as e:
        results[name] = {"Error": str(e)}

for model, metrics in results.items():
    print(f"Model: {model}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
    print()