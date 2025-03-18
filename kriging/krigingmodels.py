import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from pykrige.rk import RegressionKriging

#1. Load the data (approximations to the Pareto front)
#Files containing predictor variables
var_files = sorted(glob("VAR_*.csv"))
#Files containing target values 
fun_files = sorted(glob("FUN_*.csv"))

X = [] 
Y = [] 
coords = [] 

for var_file, fun_file in zip(var_files, fun_files):
    #Read CSV files (approximation to the Pareto front)
    #100 rows x 101 columns (Assignment)
    var_data = pd.read_csv(var_file, header=None).values
    #100 rows x 3 columns (Objective values)  
    fun_data = pd.read_csv(fun_file, header=None).values  
    p = var_data[:, :-2] 
    x = var_data[:, -2:] 
    X.append(p)
    Y.append(fun_data)
    coords.append(x)

#Convert lists to numpy arrays
X = np.vstack(X)  
Y = np.vstack(Y)
coords = np.vstack(coords).astype(np.float64)

#Display data structure
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Shape of coords:", coords.shape)

#2. Split into training and test sets
X_train, X_test, Y_train, Y_test, coords_train, coords_test = train_test_split(X, Y, coords, test_size=0.3, random_state=42)

#3. Train Regression Kriging
rk_models = [RegressionKriging(regression_model=KNeighborsRegressor(n_neighbors=5), n_closest_points=10) for _ in range(3)]
#rk_models = [RegressionKriging(regression_model=RandomForestRegressor(n_estimators=100), n_closest_points=10) for _ in range(3)]
#rk_models = [RegressionKriging(regression_model=SVR(C=0.1, gamma="auto"), n_closest_points=10) for _ in range(3)] 10e-8 ...10e8
#rk_models = [RegressionKriging(regression_model=LinearRegression(copy_X=True, fit_intercept=False), n_closest_points=10) for _ in range(3)]
#rk_models = [RegressionKriging(regression_model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3), n_closest_points=10) for _ in range(3)]

for i in range(3):
    print(f"Training RK for target {i+1}...")
    rk_models[i].fit(X_train, coords_train, Y_train[:, i])

#4. Predictions
Y_pred = np.column_stack([rk.predict(X_test, coords_test) for rk in rk_models])

#5. Compute errors
rmse = np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=0))  # RMSE per target
mae = np.mean(np.abs(Y_test - Y_pred), axis=0)  # MAE per target

print("Prediction errors:")
print(f"RMSE per target: {rmse}")
print(f"MAE per target: {mae}")
