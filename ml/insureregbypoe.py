
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1) Load data ----------
# If your file is in the same folder:
df = pd.read_csv("insurance.csv")

# ---------- 2) Define target & features ----------
target = "charges"
X = df.drop(columns=[target])
y = df[target]

# ---------- 3) Identify column types ----------
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# ---------- 4) Preprocess (impute + encode categoricals) ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)

# ---------- 5) Build pipeline: preprocessing + linear regression ----------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# ---------- 6) Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------- 7) Fit ----------
model.fit(X_train, y_train)

# ---------- 8) Predict ----------
y_pred = model.predict(X_test)

# ---------- 9) Verification metrics ----------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results (test set)")
print(f"MAE : {mae:,.4f}")
print(f"MSE : {mse:,.4f}")
print(f"RMSE: {rmse:,.4f}")
print(f"R^2 : {r2:,.6f}")

# ---------- 10) Optional: show coefficients if you want ----------
# (LinearRegression + OneHot => coefficients correspond to transformed feature space)
reg = model.named_steps["regressor"]
feature_names = []

# Build transformed feature names
ohe = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
if len(cat_cols) > 0:
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
else:
    cat_feature_names = []

feature_names = num_cols + cat_feature_names

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": reg.coef_
}).sort_values(by="coef", key=lambda s: np.abs(s), ascending=False)

print("\nTop 15 absolute coefficients:")
print(coef_df.head(15).to_string(index=False))


#If you tell me your preferred split (or cross-validation) I can adjust the code accordingly.