import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


full_dataset = pd.read_excel("./fulldataset.xlsx", index_col=False)
full_dataset[["AGE", "HEIGHT", "WEIGHT"]] = full_dataset[["AGE", "HEIGHT", "WEIGHT"]].replace(
    {" years": "", " cm": "", " kg": ""}, regex=True
).astype(float)
full_dataset[["AGE", "HEIGHT", "WEIGHT", "TIME(min)", "C-Peptide", "ISR"]] = full_dataset[["AGE", "HEIGHT", "WEIGHT", "TIME(min)", "C-Peptide", "ISR"]].astype("float64")
full_dataset["BMI"] = full_dataset["WEIGHT"] / (full_dataset["HEIGHT"]/100)**2
full_dataset["BSA"] = np.sqrt((full_dataset["HEIGHT"] * full_dataset["WEIGHT"]) / 3600)


full_dataset_cleaned = full_dataset.dropna(subset=["ISR"], how="all")

model_dataset = full_dataset_cleaned.dropna(subset=["C-Peptide", "ISR"], how="all").copy()
model_dataset = model_dataset.dropna(subset=["ISR"], how="all")
model_dataset.drop(columns=["Sample ID"], inplace=True)
model_dataset = pd.get_dummies(model_dataset, columns=["SEX", "SUBJECT"])
model_dataset = model_dataset[model_dataset["C-Peptide"] > 0] # Remove negative value outliers gotten as a result of a mistake in data collection

df = model_dataset

def create_lag_features(df, lags=4):
    df_lag = df.copy()
    for lag in range(1, lags + 1):
        df_lag[f"CPeptide_lag{lag}"] = df_lag.groupby("sample")["C-Peptide"].shift(lag)
    return df_lag.dropna()

df_lagged = create_lag_features(df)
features = ["C-Peptide", "CPeptide_lag1", "CPeptide_lag2", "sample", 
            "CPeptide_lag3", "CPeptide_lag4", 'BMI', 'BSA', 'SEX_M', 
            "SEX_F", "WEIGHT", "TIME(min)", "SUBJECT_NIDDM", "SUBJECT_normal", 
            "SUBJECT_obese"]


X_rf = df_lagged[features]
y_rf = df_lagged["ISR"]

scaling = ColumnTransformer(transformers=[
    ('scale', MinMaxScaler(), features)
])

def within_subject(selected_features):
    selected_features.remove("sample")
    X_rfr = X_rf[selected_features]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rfr, y_rf, test_size=0.2, random_state=42
    )

    return {"X_train": X_train_rf, "X_test": X_test_rf, "y_train": y_train_rf, "y_test": y_test_rf}


def outside_subject():
                    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    Xrfc = X_rf.copy()
    train_idx, test_idx = next(gss.split(Xrfc, y_rf, groups=Xrfc["sample"]))

    X_train, X_test = Xrfc.iloc[train_idx], Xrfc.iloc[test_idx]
    y_train_rf, y_test_rf = y_rf.iloc[train_idx], y_rf.iloc[test_idx]

    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    X_train_rf.drop(columns=["sample"],inplace=True)

    X_test_rf.drop(columns=["sample"],inplace=True)

    return {"X_train": X_train_rf, "X_test": X_test_rf, "y_train": y_train_rf, "y_test": y_test_rf}



def model_selection():
    linear_model = LinearRegression()
    xgboost_model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, subsample=1.0, colsample_bytree=0.9, reg_alpha=0.8)
    random_forest = RandomForestRegressor(n_estimators=200, max_depth=10, max_features=0.99, max_samples=0.92, random_state=42)


    base_model = [("xgboost", xgboost_model),('randomforest', random_forest)]
    meta_model = LinearRegression()
    stacked_model = StackingRegressor(estimators=base_model, final_estimator=meta_model, passthrough=False)
    voting_regressor = VotingRegressor(estimators=base_model)

    model_inputs = {"1": linear_model, "2": xgboost_model, "3": random_forest, "4": stacked_model, "5": voting_regressor}

    model_input = input('Please select a model [1: linear_model, 2: xgboost_model, 3: random_forest, 4: stacked_model, 5: voting_regressor]: ')
    rf_model = Pipeline(steps=[
            ("scale", scaling),
            ("model", model_inputs[model_input])
    ])

    return {"selected_model": rf_model, "model_inputs": model_inputs}




def model_results(splitting_style):

    model_selection()["selected_model"].fit(splitting_style["X_train"], splitting_style["y_train"])
    rf_pred = model_selection()["selected_model"].predict(splitting_style["X_test"])


    print("Ensemble Voting Regressor RMSE:", np.sqrt(mean_squared_error(splitting_style["y_test"], rf_pred)))
    print(f'Ensemble Voting Regressor R2_Score: {r2_score(splitting_style["y_test"], rf_pred)}')
    print(f'Ensemble Voting Regressor Mean Absolute Error {mean_absolute_error(splitting_style["y_test"], rf_pred)}')


    r, p_value = pearsonr(splitting_style["y_test"], rf_pred)

    print("Pearson correlation:", r)
    print("p-value:", p_value)


    return splitting_style["y_test"].max(), splitting_style["y_test"].min()

print(model_results(within_sub))