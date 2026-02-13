import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr



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
model_dataset = model_dataset[model_dataset["C-Peptide"] > 0]


label = model_dataset["ISR"]
model_features = model_dataset.copy()
model_features.drop(columns=["ISR"], inplace=True)

model_train, model_test, result_train, result_test = train_test_split(model_features, label, test_size=0.2, random_state=42)


model_features = pd.concat([model_train, result_train], axis=1)
model_test_features = pd.concat([model_test, result_test], axis=1)
model_features

class WithinSubjectDataset:
    def __init__(self, df1, df2, label_col="ISR", lags=3):
        self.df = df1.copy()
        self.df2 = df2.copy()
        self.label_col = label_col
        self.lags = lags

    def create_lag_features(self, df):
        df_lag = df.copy()
        for lag in range(1, self.lags + 1):
            df_lag[f"CPeptide_lag{lag}"] = df_lag.groupby("sample")["C-Peptide"].shift(lag)
        return df_lag.dropna()

    def prepare(self):
        df_train_lagged = self.create_lag_features(self.df)
        df_test_lagged = self.create_lag_features(self.df2)
        features = df_train_lagged.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
        to_remove = ["ISR", "AGE", "HEIGHT", "WEIGHT"]
        features = [x for x in features if x not in to_remove]
        
        return {
            "X_train": df_train_lagged[features],
            "X_test": df_test_lagged[features],
            "y_train": np.log1p(df_train_lagged[self.label_col]),
            "y_test": np.log1p(df_test_lagged[self.label_col]),
        }


class OutsideSubjectDataset:
    def __init__(self, df, label_col="ISR"):
        self.df = df.copy()
        self.label_col = label_col

    def prepare(self):

        y = np.log1p(self.df[self.label_col])
        X = self.df.drop(columns=[self.label_col])

        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=X["sample"]))

        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

        # Concatenate for lag creation
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        df_train_lagged = train_df
        df_test_lagged = test_df
        features = df_train_lagged.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
        to_remove = ["ISR", "AGE", "HEIGHT", "WEIGHT"]

        features = [x for x in features if x not in to_remove]

        return {
            "X_train": df_train_lagged[features],
            "X_test": df_test_lagged[features],
            "y_train": df_train_lagged[self.label_col],
            "y_test": df_test_lagged[self.label_col],
        }


class RegressorEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate(self, model, scale_numeric=True):
        if scale_numeric:
            numeric_features = self.X_train.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
            scaler = ColumnTransformer([('num', MinMaxScaler(), numeric_features)])
            pipeline = Pipeline([
                ("scale", scaler),
                ("model", model)
            ])
        else:
            pipeline = model

        pipeline.fit(self.X_train, self.y_train)
        y_pred = np.expm1(pipeline.predict(self.X_test))
        y_true = np.expm1(self.y_test)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r, p = pearsonr(y_true, y_pred)

        print(f"Model: {model.__class__.__name__}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"Pearson r: {r:.3f}, p-value: {p:.2e}")
        rho, p = spearmanr(y_true, y_pred)
        print(f'Spearman = {rho, p}')


        return {"rmse": rmse, "r2": r2, "mae": mae, "pearson_r": r, "p_value": p}



# Prepare datasets
within_subjects = WithinSubjectDataset(model_features, model_test_features).prepare()
cross_subjects = OutsideSubjectDataset(model_dataset).prepare()


linear_model = LinearRegression()
xgboost_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.2, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.8)
random_forest = RandomForestRegressor(n_estimators=300, max_depth=None, max_features=0.7, random_state=42, min_samples_split=5)

base_model = [("xgboost", xgboost_model),('randomforest', random_forest)]

meta_model = LinearRegression()

voting_regressor = VotingRegressor(estimators=base_model)

evaluator = RegressorEvaluator(**within_subjects)
metrics = evaluator.evaluate(voting_regressor)


explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(within_subjects["X_test"])

shap.summary_plot(shap_values, within_subjects["X_test"])

