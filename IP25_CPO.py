# Constrained Portfolio Optimisation
# pc = personal
# env: PROD308_C_N_TENSOR

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#from itertools import product
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
import time
import logging
import missingno

#############################################
# PROCEDURES
#############################################

def log_and_print(message):
    print(message)
    logging.info(message)

def Pause_Code():
    import sys
    print()
    print()
    stop = input('Type Y to go ahead, N to stop code: ')
    if stop == 'N':
        sys.exit()

# Rolling Sharpe Ratio calculation
def rolling_sharpe_ratio(returns, window=30, risk_free_rate=0.0):
    """
    Calculate rolling Sharpe ratio:
    SR = (mean(portfolio_returns - risk_free_rate)) / std(portfolio_returns)
    """
    excess_returns = returns - risk_free_rate
    rolling_mean = pd.Series(excess_returns).rolling(window).mean()
    rolling_std = pd.Series(excess_returns).rolling(window).std()
    sharpe_ratio = rolling_mean / rolling_std

    # Replace infinities with 0
    sharpe_ratio.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Ensure no NaN values
    sharpe_ratio.fillna(0, inplace=True)

    return sharpe_ratio

# Function to evaluate model based on user-selected metric
def evaluate_model(name, model, X_test, y_test, metric):
    log_and_print(f"\nEvaluating {name} Model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Directional Accuracy
    y_pred_dir = np.sign(y_pred)
    y_test_dir = np.sign(y_test)
    directional_accuracy = np.mean(y_pred_dir == y_test_dir)

    # Display results based on selected metric
    log_and_print(f"{name} Model Results:")
    log_and_print(f"R²: {r2:.4f}")
    log_and_print(f"MSE: {mse:.4f}")
    log_and_print(f"MAE: {mae:.4f}")
    log_and_print(f"Directional Accuracy: {directional_accuracy:.4f}")
    log_and_print("-" * 40)
    
    # Return selected metric score
    if metric == "R²":
        return r2
    elif metric == "MAE":
        return mae
    elif metric == "DA":
        return directional_accuracy
    else:
        raise ValueError(f"Invalid metric selected: {metric}")

def time_grouped_cv(model, X, y):
    
    # cross validation
    
    kf = KFold(n_splits=5, shuffle=False)
    scores = []
    
    combined_matrix_filtered = combined_matrix.loc[X.index].copy()
    combined_matrix_filtered['time_step'] = combined_matrix_filtered['time_step'].reset_index(drop=True)
    
    print("\nStarting Time-Grouped Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.unique(combined_matrix_filtered['time_step'])), start=1):
        log_and_print(f"Processing Fold {fold}/5...")
        time.sleep(0.2)  # Just to make it more readable if running fast
        
        train_steps = combined_matrix_filtered['time_step'].iloc[train_idx].values
        test_steps = combined_matrix_filtered['time_step'].iloc[test_idx].values
        
        train_mask = combined_matrix_filtered['time_step'].isin(train_steps).values
        test_mask = combined_matrix_filtered['time_step'].isin(test_steps).values       
        
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]        
                
        model.fit(X_train, y_train)
        
        #^^^^^^^^^^^^^^^^
        y_pred = model.predict(X_test)
        
        # Calculate selected metric
        score = selected_metric_func(y_test, y_pred)
        scores.append(score)
        log_and_print(f"Fold {fold} {selected_metric_name}: {score:.4f}")
        
    mean_score = np.mean(scores)
    # log_and_print(f"Completed Cross-Validation. Mean R²: {mean_score:.4f}\n")
    log_and_print(f"Completed Cross-Validation. Mean {selected_metric_name}: {mean_score:.4f}\n")
    return mean_score


def custom_random_search(model, param_grid, X, y, n_iter=10):
    best_score = -np.inf
    best_params = None
    feature_columns = X.columns.tolist()  # Get the feature names directly from X
        
    log_and_print("\nStarting Randomized Search...")
    log_and_print(f'Model = {model}')
    
    for i in range(n_iter):
        log_and_print(f"Iteration {i+1}/{n_iter}...")
        params = {k: np.random.choice(v) for k, v in param_grid.items()}
        model.set_params(**params)
        
        score = time_grouped_cv(model, X, y)
        log_and_print(f"Iteration {i+1} Score: {score:.4f}, Params: {params}\n")
        
        if score > best_score:
            best_score = score
            best_params = params
    
    log_and_print(f"Randomized Search Completed for model = {model}")
    log_and_print(f"Best Score: {best_score:.4f}")
    log_and_print(f"Best Params: {best_params}\n")
    
    model.set_params(**best_params)    
    model.fit(X, y)
    # Feature Importance Extraction
    if hasattr(model, "feature_importances_"):
        plot_feature_importances(model, feature_columns, model.__class__.__name__)
    return model, best_score, best_params


def add_engineered_features(df):
   
    
    # Inflation differential
    df["inflation_diff"] = df["INFLEUR5Y"] - df["INFLUSD5Y"]
    
    # Volatility spread
    df["VIX_V2X_spread"] = df["VIX"] - df["V2X"]
    
    # Yield spreads
    df["EUR_yield_spread"] = df["5Y EUR RATE"] - df["3M EUR RATE"]
    df["USD_yield_spread"] = df["5Y USD RATE"] - df["3M USD RATE"]
    
    # FX spread
    df["EURUSD_EURJPY_spread"] = df["EURUSD"] - df["EURJPY"]
    
    # Commodity spread
    df["GOLD_WTI_spread"] = df["GOLD"] - df["WTI"]
    
    # Differenced features
    for col in ["VIX", "V2X", "MOVE", "EURUSD", "EURJPY", "GOLD", "5Y EUR RATE", "inflation_diff"]:
        df[f"{col}_diff"] = df[col].diff()
    
    # Interaction terms
    df["VIX_MOVE_interaction"] = df["VIX_diff"] * df["MOVE_diff"]
    df["VIX_EURUSD_interaction"] = df["VIX_diff"] * df["EURUSD_diff"]
    
    # Momentum Indicators: Rolling Averages (7-day and 30-day)
    for col in df.columns:
        if col.endswith('_diff'):
            df[f'{col}_ma7'] = df[col].rolling(window=7).mean()
            df[f'{col}_ma30'] = df[col].rolling(window=30).mean()

    # Volatility Clustering: Rolling Standard Deviations
    for col in ['VIX', 'V2X', 'MOVE']:
        df[f'{col}_volatility_7d'] = df[col].rolling(window=7).std()
        df[f'{col}_volatility_30d'] = df[col].rolling(window=30).std()

    # Non-linear Effects: Squared Terms
    for col in df.columns:
        if col.endswith('_diff'):
            df[f'{col}_squared'] = df[col] ** 2

    # Signal Ratios
    df['VIX_MOVE_ratio'] = df['VIX'] / df['MOVE'].replace(0, np.nan)
    df['EURUSD_VIX_ratio'] = df['EURUSD'] / df['VIX'].replace(0, np.nan)
    df['EUR_Inflation_Rate_ratio'] = df['5Y EUR RATE'] / df['INFLUSD5Y'].replace(0, np.nan)
    df['USD_Rate_Spread_ratio'] = df['5Y USD RATE'] / df['3M USD RATE'].replace(0, np.nan)
    
    # Drop NaN rows after diff()
    df.dropna(inplace=True)
    
    return df


def plot_feature_importances(model, feature_names, model_name):
    
    """Plots and prints the top 10 most important features."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)

    # Print the top 10 features
    log_and_print(f"\nTop 10 Important Features for {model_name}:")
    for i, row in feature_importance_df.iterrows():
        log_and_print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Plot the feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="steelblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - {model_name}")
    plt.gca().invert_yaxis()
    plt.show()
    
def directional_accuracy(y_true, y_pred):
    """Calculates the percentage of times the predicted and actual returns have the same sign."""
    correct_directions = (np.sign(y_true) == np.sign(y_pred)).sum()
    return correct_directions / len(y_true)

#############################################
# END PROCEDURES
#############################################

#############################################
# PARAMETERS GRID
#############################################

# Define hyperparameter grids

gbr_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10, 20, 30],    # Added regularization
    'min_samples_leaf': [1, 2, 4]               # Added regularization
}

xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],              # Added regularization
    'gamma': [0, 0.1, 0.2],                     # Added regularization
    'reg_alpha': [0, 0.1, 0.5],                 # L1 regularization (lasso)
    'reg_lambda': [1, 1.5, 2.0, 3.0]            # L2 regularization (ridge)
}

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],           # Added regularization
    'max_samples': [0.5, 0.75, 1.0]             # Added regularization
}

###########################################

# parameters
n_assets = 4
N = 500  # combinations
rnd_seed = 42
date_start = '2020-01-30'

# User-selectable evaluation metric
print()
print("Select evaluation metric (type the corresponding number):")
print()
print("1: R-squared (R²)")
print("2: Mean Absolute Error (MAE)")
print("3: Directional Accuracy (DA)")
print()
metric_choice = input("Enter choice (1/2/3): ")

# Map choice to metric
metric_map = {
    "1": ("R²", r2_score),
    "2": ("MAE", mean_absolute_error),
    "3": ("DA", directional_accuracy)
}

# Get selected metric
if metric_choice not in metric_map:
    print()
    print("Invalid choice. Defaulting to R².")
    selected_metric_name, selected_metric_func = metric_map["1"]
else:
    selected_metric_name, selected_metric_func = metric_map[metric_choice]
print()
print(f"Selected metric: {selected_metric_name}")

# === Set up logging ===
log_file = "results_log.txt"
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load real data
filename = '/Users/matteoprandi/Documents/QF UniBo/Advanced Topics in Artificial Intelligence/IP25_DATA_day1/CPO_DATA.xlsx'
data = pd.read_excel(filename, header=3, sheet_name='RAW_DATA')
data.set_index('date', inplace=True)

######################
# setting final sample
######################
data = data[date_start:]

missingno.matrix(data.iloc[:,0:4],figsize=(10,5), fontsize=12)
plt.title("ASSETS' RETURNS")
plt.show()

missingno.matrix(data.iloc[:,4:],figsize=(10,5), fontsize=12)
plt.title("MARKET FACTORS")
plt.show()

data = add_engineered_features(data)

missingno.matrix(data,figsize=(10,5), fontsize=12)
plt.title("FULL FEATURES")
plt.show()

Pause_Code()

# Calculate returns for the first 4 columns (assets) and drop NaN rows
data.iloc[:, 0:n_assets] = data.iloc[:, 0:n_assets].pct_change()
data = data.dropna()

# Extract asset returns (4 assets) and features (15 features)
asset_returns = data.iloc[:, 0:n_assets]
features = data.iloc[:, n_assets:]

# Generate random weight combinations using Dirichlet distribution
np.random.seed(rnd_seed)
random_weights = np.random.dirichlet(np.ones(n_assets), size=N)

# Convert to DataFrame
portfolio_weights = pd.DataFrame(random_weights, columns=['w1', 'w2', 'w3', 'w4'])

print("Sample of random portfolio weights:")
print(portfolio_weights.head())

# Number of time steps (T) and portfolios (N)
T = len(features)
N = len(portfolio_weights)

print(f"Number of time steps (T): {T}")
print(f"Number of portfolio combinations (N): {N}")

# Create the correct NxT matrix
# 1. Repeat each row of features N times
features_repeated = features.loc[features.index.repeat(N)].reset_index(drop=True)

# 2. Tile (cycle) portfolio weights T times so they align with features
portfolio_repeated = pd.DataFrame(
    np.tile(portfolio_weights, (T, 1)),
    columns=['w1', 'w2', 'w3', 'w4']
)

# 3. Combine features and weights side by side
combined_matrix = pd.concat([portfolio_repeated, features_repeated], axis=1)

print("Sample of combined NxT matrix:")
print(combined_matrix.head(10))
print(f"Shape of the final matrix: {combined_matrix.shape}")

# Repeat asset returns N times to match the shape of portfolio_repeated
asset_returns_repeated = pd.concat([asset_returns] * N, ignore_index=True)

# Ensure shapes match: (NxT, 4) @ (NxT, 4) -> (NxT, 1)
print("Shape of portfolio_repeated:", portfolio_repeated.shape)
print("Shape of asset_returns_repeated:", asset_returns_repeated.shape)

# Recalculate portfolio returns using weights and asset returns
portfolio_returns = np.sum(portfolio_repeated.values * asset_returns.loc[asset_returns.index.repeat(N)].values, axis=1)

# Convert to DataFrame for consistency
portfolio_returns_df = pd.DataFrame(portfolio_returns, columns=['portfolio_return'])

# Add portfolio returns to the combined matrix
combined_matrix['portfolio_return'] = portfolio_returns_df['portfolio_return'].values

print("Sample of combined matrix with portfolio returns:")
print(combined_matrix.head(10))

# Calculate Sharpe ratios for all rows (NxT)
combined_matrix['sharpe_ratio'] = rolling_sharpe_ratio(combined_matrix['portfolio_return'], window=30)

# Handle NaNs by replacing with zero (or discuss a better approach if needed)
combined_matrix['sharpe_ratio'].fillna(0, inplace=True)

print("Sample of combined matrix with Sharpe ratios:")
print(combined_matrix.head(10))
print(f"Shape of the final matrix: {combined_matrix.shape}")

# Remove rows where Sharpe ratio is NaN or zero
combined_matrix = combined_matrix[combined_matrix['sharpe_ratio'] != 0].reset_index(drop=True)

print("Sample of cleaned matrix (no zero Sharpe ratios):")
print(combined_matrix.head(10))
print(f"New shape of the matrix: {combined_matrix.shape}")

# Group rows by time steps
time_steps = np.arange(len(features))

# Assign time step groups, repeated N times (so rows for t1-tN are grouped)
combined_matrix['time_step'] = np.repeat(time_steps, N)[:len(combined_matrix)]

missingno.matrix(combined_matrix,figsize=(10,5), fontsize=12)
plt.title("COMBINED MATRIX")
plt.show()

# PREDICTION MATRIX
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
N_P = 2 # how many time step to predict from bottom
predict_matrix = combined_matrix[(-N*N_P):]
combined_matrix = combined_matrix[0:(-N*N_P)]
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

missingno.matrix(combined_matrix,figsize=(10,5), fontsize=12)
plt.title("COMBINED MATRIX")
plt.show()

missingno.matrix(predict_matrix,figsize=(10,5), fontsize=12)
plt.title("PREDICTION MATRIX")
plt.show()

print()
print('matrix preparation DONE')
Pause_Code()

##############################
############### MODEL TRAINING
##############################

# Prepare features (X) and target (y)
X = combined_matrix.drop(columns=['sharpe_ratio', 'portfolio_return', 'time_step'])
y = combined_matrix['sharpe_ratio']

# Ensure indices of X and y match
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Matching indices: {(X.index == y.index).all()}")

split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

gbr = GradientBoostingRegressor(random_state=rnd_seed)
rf = RandomForestRegressor(random_state=rnd_seed)
xgb = XGBRegressor(random_state=rnd_seed)

# Random search with time grouping
rf_best_model, rf_best_score, rf_best_params = custom_random_search(rf, rf_params, X_train, y_train)
gbr_best_model, gbr_best_score, gbr_best_params = custom_random_search(gbr, gbr_params, X_train, y_train)
xgb_best_model, xgb_best_score, xgb_best_params = custom_random_search(xgb, xgb_params, X_train, y_train)


print("\nBest Random Forest hyperparameters:", rf_best_params)
print("Regularization params (RF): max_features={}, max_samples={}".format(
    rf_best_params.get('max_features'), rf_best_params.get('max_samples')))

print("\nBest Gradient Boosting hyperparameters:", gbr_best_params)
print("Regularization params (GBR): min_samples_split={}, min_samples_leaf={}".format(
    gbr_best_params.get('min_samples_split'), gbr_best_params.get('min_samples_leaf')))

print("\nBest XGBoost hyperparameters:", xgb_best_params)
print("Regularization params (XGB): min_child_weight={}, gamma={}, reg_alpha={}, reg_lambda={}".format(
    xgb_best_params.get('min_child_weight'), 
    xgb_best_params.get('gamma'),
    xgb_best_params.get('reg_alpha'),
    xgb_best_params.get('reg_lambda')))

_ = evaluate_model("Random Forest", rf_best_model, X_test, y_test,selected_metric_name)
_ = evaluate_model("Gradient Boosting", gbr_best_model, X_test, y_test,selected_metric_name)
_ = evaluate_model("XGBoost", xgb_best_model, X_test, y_test,selected_metric_name)

# === Properly close log file ===
logger.removeHandler(file_handler)
file_handler.close()

# prediction
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
X_pred = predict_matrix.drop(columns=['sharpe_ratio', 'portfolio_return', 'time_step'])

yhat_rf = rf_best_model.predict(X_pred)
yhat_gbr = gbr_best_model.predict(X_pred)
yhat_xgb = xgb_best_model.predict(X_pred)

X_pred['yhat_rf'] = yhat_rf
X_pred['yhat_gbr'] = yhat_gbr
X_pred['yhat_xgb'] = yhat_xgb

W_rf = X_pred[X_pred['yhat_rf'] == X_pred['yhat_rf'].max()].iloc[:,0:4]
W_gbr = X_pred[X_pred['yhat_gbr'] == X_pred['yhat_gbr'].max()].iloc[:,0:4]
W_xgb = X_pred[X_pred['yhat_xgb'] == X_pred['yhat_xgb'].max()].iloc[:,0:4]






