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

model_dataset = full_dataset_cleaned.dropna(subset=["C-Peptide"], how="all").copy()
model_dataset.drop(columns=["Sample ID"], inplace=True)
model_dataset = pd.get_dummies(model_dataset, columns=["SEX", "SUBJECT"])
model_dataset = model_dataset[model_dataset["C-Peptide"] > 0] # Remove negative value outliers gotten as a result of a mistake in data collection


c_pep_features = ["ISR", "BMI", "WEIGHT", "BSA"]
preprocessing = ColumnTransformer(transformers=[
    ("scale", MinMaxScaler(), c_pep_features)
])
c_pep_none = full_dataset_cleaned[full_dataset_cleaned["C-Peptide"].isna()]
y = np.log1p(model_dataset["C-Peptide"]) # This converts large C-Peptide values to log for better scalability
X = model_dataset[c_pep_features]

X_test = c_pep_none[c_pep_features]

model_pipeline = Pipeline(steps=[
    ("scale", preprocessing),
    ("model", XGBRegressor(n_estimators=400, max_depth=10, learning_rate=0.05, subsample=1))
])

model_pipeline.fit(X, y)
predictions = model_pipeline.predict(X_test)
predictions = np.expm1(predictions) # converts the log predicted value back to base 10
c_pep = pd.DataFrame(predictions)
c_pep.rename(columns={0: "C-Peptide"}, inplace=True)
c_pep_none = pd.get_dummies(c_pep_none, columns=["SUBJECT", "SEX"])
replaced = c_pep_none.copy()
replaced.drop(columns=["C-Peptide"], inplace=True)
replaced = replaced.reset_index(drop=True)
final_cpep = pd.concat([replaced, c_pep], axis=1)


real_dataset = pd.concat([model_dataset, final_cpep])
real_model_dataset = real_dataset.copy()
real_model_dataset.drop(columns=["Sample ID"], inplace=True)
df = real_model_dataset

def create_lag_features(df, lags=4):
    df_lag = df.copy()
    for lag in range(1, lags + 1):
        df_lag[f"CPeptide_lag{lag}"] = df_lag.groupby("sample")["C-Peptide"].shift(lag)
    return df_lag.dropna()

df_lagged = create_lag_features(df)
features = ["C-Peptide", "CPeptide_lag1", "CPeptide_lag2", "sample", 
            "CPeptide_lag3", "CPeptide_lag4", 'BMI', 'BSA', 'SEX_M', 
            "SEX_F", "WEIGHT", "TIME(min)", "SUBJECT_NIDDM", "SUBJECT_normal", 
            "SUBJECT_obese", "sample"]
X_rf = df_lagged[features]
y_rf = df_lagged["ISR"]
X_rf


scaling = ColumnTransformer(transformers=[
    ('scale', MinMaxScaler(), features)
])

def within_subject(selected_features):
    selected_features.remove("sample")
    X_rf = X_rf[selected_features]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42
    )

    return {"X_train": X_train_rf, "X_test": X_test_rf, "y_train": y_train_rf, "y_test": y_test_rf}


def outside_subject():
                    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X_rf, y_rf, groups=X_rf["sample"]))

    X_train, X_test = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
    y_train_rf, y_test_rf = y_rf.iloc[train_idx], y_rf.iloc[test_idx]

    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    X_train_rf.drop(columns=["sample"],inplace=True)

    X_test_rf.drop(columns=["sample"],inplace=True)

    return {"X_train": X_train_rf, "X_test": X_test_rf, "y_train": y_train_rf, "y_test": y_test_rf}



linear_model = LinearRegression()
xgboost_model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, subsample=1.0, colsample_bytree=0.9, reg_alpha=0.8)
random_forest = RandomForestRegressor(n_estimators=200, max_depth=10, max_features=0.99, max_samples=0.92, random_state=42)

base_model = [("xgboost", xgboost_model),('randomforest', random_forest)]

meta_model = LinearRegression()

stacked_model = StackingRegressor(estimators=base_model, final_estimator=meta_model, passthrough=False)
voting_regressor = VotingRegressor(estimators=base_model)
rf_model = Pipeline(steps=[
        ("scale", scaling),
        ("model", stacked_model)
])

within_sub = within_subject(features)
outside = outside_subject()


def model_results(splitting_style):

    rf_model.fit(splitting_style["X_train"], splitting_style["y_train"])
    rf_pred = rf_model.predict(splitting_style["X_test"])

    print("Ensemble Voting Regressor RMSE:", np.sqrt(mean_squared_error(splitting_style["y_test"], rf_pred)))
    print(f'Ensemble Voting Regressor R2_Score: {r2_score(splitting_style["y_test"], rf_pred)}')
    print(f'Ensemble Voting Regressor Mean Absolute Error {mean_absolute_error(splitting_style["y_test"], rf_pred)}')


    r, p_value = pearsonr(splitting_style["y_test"], rf_pred)

    print("Pearson correlation:", r)
    print("p-value:", p_value)


    return splitting_style["y_test"].max(), splitting_style["y_test"].min()