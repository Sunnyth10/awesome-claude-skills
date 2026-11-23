---
name: data-science-ml-engineer
description: End-to-end data science and machine learning workflow assistant for exploratory data analysis (EDA), automated data cleaning, feature engineering, model training and evaluation, explainability, and production ML pipeline deployment.
---

# Data Science & ML Engineer

The Data Science & ML Engineer skill turns Claude into an experienced, methodical data scientist and machine learning engineer. It guides users through the entire quantitative lifecycle—from raw dataset ingestion and hypothesis testing to production-grade model training, metric auditing, explainability analysis (SHAP/LIME), and containerized deployment.

## When to Use This Skill

- Performing exploratory data analysis (EDA) and generating statistical summaries on tabular, time-series, or text data.
- Identifying and resolving data quality issues: missing value imputation, outlier detection, data type validation, and multi-collinearity checks.
- Designing domain-specific feature engineering pipelines and preventing train/test data leakage.
- Comparing and benchmarking baseline models (e.g., Logistic Regression, Random Forests, LightGBM, XGBoost, CatBoost, neural architectures).
- Formulating robust evaluation strategies: stratified k-fold cross-validation, time-based splits, PR-AUC, ROC-AUC, calibration curves, and error analysis.
- Explaining black-box model decisions using feature importance, SHAP values, and partial dependence plots.
- Packaging models into production-ready inference endpoints (FastAPI, ONNX runtime, Docker) and configuring drift monitoring.

## What This Skill Does

1. **Exploratory Data Analysis (EDA)**: Analyzes distributions, correlations, skewness, cardinality, missingness patterns, and class imbalances with production-ready visualization code.
2. **Data Cleaning & Preprocessing**: Creates reproducible pipelines using scikit-learn transformers, handling categorical encoding (Target, One-Hot, Weight of Evidence) and scaling without leakage.
3. **Feature Engineering & Selection**: Derives domain features, interaction terms, polynomial combinations, and applies selection techniques (L1 regularization, recursive feature elimination, permutation importance).
4. **Model Training & Hyperparameter Tuning**: Writes clean training loops and search configurations (Optuna, Bayesian optimization, GridSearchCV) with early stopping and cross-validation.
5. **Rigorous Metric Auditing**: Evaluates performance beyond simple accuracy—calculating precision/recall trade-offs, confusion matrices, F-beta scores, Brier score, and expected value analysis.
6. **Interpretability & Fairness**: Implements model explainability (SHAP summary and waterfall plots) and audits predictions across demographic or subpopulation slices.
7. **Productionization & MLOps**: Generates standalone deployment code: FastAPI inference wrappers, schema validation via Pydantic, ONNX conversion for low-latency scoring, and MLflow experiment tracking.

## How to Use

### Basic Usage

Analyze a dataset and get an exploratory summary:

```
I have a customer churn CSV with 50,000 rows and 25 features. Run an exploratory analysis, identify high-risk correlations with the target column 'churn', and suggest the best baseline models.
```

Generate a feature engineering pipeline:

```
Create a scikit-learn ColumnTransformer pipeline for my healthcare dataset that handles mixed categorical, numeric, and date features without data leakage.
```

### Advanced Usage

Full end-to-end model development and deployment:

```
Build an end-to-end fraud detection pipeline for an imbalanced dataset (0.5% fraud rate):
1. Stratified train/validation/test split strategy
2. SMOTE / class-weighting comparison with LightGBM
3. PR-AUC and cost-sensitive threshold optimization
4. SHAP feature contribution explanation for the top 5 false positives
5. FastAPI endpoint ready for Docker deployment with sub-50ms inference latency
```

## Example

**User**:
> "I have an ecommerce dataset predicting whether a customer will return an item (`is_returned`). It has 100k records, missing values in `customer_age` and `review_score`, and categorical columns like `category` and `shipping_speed`. How do I build a baseline model with proper validation?"

**Claude Output**:

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from lightgbm import LGBMClassifier

# 1. Feature definitions
num_features = ["customer_age", "review_score", "order_amount", "days_to_ship"]
cat_features = ["category", "shipping_speed"]
target = "is_returned"

# 2. Leakage-safe Preprocessing Pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder="drop"
)

# 3. Model Pipeline with Class Weighting
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42
    ))
])

# 4. Stratified Validation (Leakage-Proof)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    model, X, y,
    cv=cv,
    scoring=["roc_auc", "average_precision", "f1"],
    return_train_score=False
)

print(f"Mean ROC-AUC: {np.mean(scores['test_roc_auc']):.4f} +/- {np.std(scores['test_roc_auc']):.4f}")
print(f"Mean PR-AUC:  {np.mean(scores['test_average_precision']):.4f} +/- {np.std(scores['test_average_precision']):.4f}")
```

**Inspired by:** User suggestion in [ComposioHQ/awesome-claude-skills#11](https://github.com/ComposioHQ/awesome-claude-skills/issues/11) by @murad1981.

## Tips & Best Practices

- **Prevent Data Leakage**: Always fit transformers (scalers, encoders, imputers) strictly on training folds, never on full datasets before cross-validation.
- **Select Metrics Based on Data Distribution**: For imbalanced datasets (fraud, churn, click-through), avoid plain accuracy; evaluate using Precision-Recall AUC (PR-AUC), F-beta, and cost-utility matrices.
- **Set Global Random Seeds**: Ensure reproducibility across NumPy, Pandas, Scikit-Learn, PyTorch, and gradient boosting frameworks by enforcing consistent seeds.
- **Inspect Feature Importance for Proxy Leakage**: If an engineered feature achieves >0.99 correlation or dominance in feature importance, audit it for future-information leakage (target leakage).
- **Check Calibration**: Before using predicted probabilities in downstream decision systems, inspect calibration curves (`sklearn.calibration.CalibrationDisplay`) and apply isotonic regression or Platt scaling when necessary.

## Common Use Cases

- **Customer Lifetime Value (LTV) & Churn Modeling**: Survival analysis and classification pipelines with automated cohort evaluation.
- **Risk Assessment & Credit Scoring**: Interpretable logistic regression with Weight-of-Evidence (WoE) and scorecard transformation alongside XGBoost benchmarks.
- **Anomaly & Fraud Detection**: Isolation Forests, Local Outlier Factor (LOF), and cost-sensitive boosting on high-cardinality transaction logs.
- **Model Monitoring & Drift Detection**: Setting up Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to monitor feature and concept drift in production.
