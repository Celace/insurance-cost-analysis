# Insurance Cost Prediction – Linear & Ridge Regression

**Tech Stack:** `Python`, `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

Predicts annual medical-insurance charges using patient attributes (age, gender, BMI, children, smoker status, region).  
Achieved **R² = 0.85** (full-data polynomial pipeline) and **R² = 0.78** on unseen test data with Ridge regression.

---

## Project Overview

| Goal | Result |
|------|--------|
| Clean & explore the dataset | 2,772 rows, missing `?` → `NaN`, imputed |
| Identify strongest predictor | `smoker` (correlation ≈ 0.79) |
| Baseline linear model (only `smoker`) | **R² = 0.62** |
| Multi-variable linear model | **R² = 0.75** |
| Polynomial + scaling pipeline | **R² = 0.85** |
| Ridge (α = 0.1) on 80/20 split | **R² = 0.78** (test) |

---

## Step-by-Step Workflow

### 1. Data Acquisition
```python
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv"
df = pd.read_csv(url, header=None)
```

### 2. Add Column Headers
```python
pythonheaders = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers
```

### 3. Handle Missing Values
```python
pythondf.replace("?", np.nan, inplace=True)

# smoker (categorical) → most frequent value
df["smoker"].fillna(df["smoker"].mode()[0], inplace=True)

# age (continuous) → mean
mean_age = df["age"].astype(float).mean()
df["age"].fillna(mean_age, inplace=True)

df[["age", "smoker"]] = df[["age", "smoker"]].astype(int)
```
### 4. Exploratory Data Analysis (EDA)
```python
pythondf.info()
df.describe()
sns.boxplot(x="smoker", y="charges", data=df)
print(df.corr())
Key Insight: smoker has the highest correlation with charges (~0.79).
```
### 5. Baseline Model – Smoker Only
```python
pythonX = df[["smoker"]]
y = df["charges"]
lm = LinearRegression()
lm.fit(X, y)
print(lm.score(X, y))  # → 0.622
```

### 6. Multi-Variable Linear Regression
```python
pythonZ = df.drop("charges", axis=1)
lm.fit(Z, y)
print(lm.score(Z, y))  # → 0.750
```

### 7. Polynomial + Scaling Pipeline
```python
pythonfrom sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("poly",  PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression())
])

pipe.fit(Z, y)
y_pred = pipe.predict(Z)
print(r2_score(y, y_pred))  # → 0.845
```

### 8. Train-Test Split & Ridge Regression
```python
pythonfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(
    Z, y, test_size=0.2, random_state=1
)

pr = PolynomialFeatures(degree=2)
X_train_pr = pr.fit_transform(X_train)
X_test_pr  = pr.transform(X_test)

ridge = Ridge(alpha=0.1)
ridge.fit(X_train_pr, y_train)
y_hat = ridge.predict(X_test_pr)
print(r2_score(y_test, y_hat))  # → 0.784
```

# Files in This Repository

insurance-cost-analysis.ipynb – Full Jupyter notebook with code, outputs, and visualizations
insurance.csv – Cleaned dataset (optional – can be re-downloaded from the URL)
README.md – This document


# How to Run
bash 
# 1. Clone the repo
git clone https://github.com/<your-username>/insurance-cost-analysis.git
cd insurance-cost-analysis

# 2. Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib

# 3. Launch notebook
jupyter notebook "insurance-cost-analysis.ipynb"
