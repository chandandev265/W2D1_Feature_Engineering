
import pandas as pd
import numpy as np

np.random.seed(7)

n = 80

df = pd.DataFrame({
    'age': np.random.randint(19, 28, n),
    'study_hrs': np.random.uniform(1, 8, n).round(1),
    'midterm': np.random.randint(30, 100, n),
    'assignments': np.random.randint(35, 100, n),
    'attendance': np.random.randint(50, 101, n),
    'study_mode': np.random.choice(['Online', 'Offline'], n),
    'performance': np.random.choice(['Low', 'Medium', 'High'], n)
})

df['final_score'] = (
    df['midterm'] * 0.35 +
    df['assignments'] * 0.25 +
    df['study_hrs'] * 3 +
    df['attendance'] * 0.15 +
    np.random.normal(0, 5, n)
).clip(0, 100).round(1)

print("=== DATASET PREVIEW ===")
print(df.head())

print("\n=== DATASET INFORMATION ===")
print(df.info())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe().round(2))

print("\n=== TARGET DISTRIBUTION ===")
print(df['performance'].value_counts())

print("\n=== PASS RATE ===")
print(f"{(df['final_score'] >= 60).mean():.1%}")

df.to_csv("data/student_dataset.csv", index=False)

print("\nDataset saved successfully!")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

print("\n========== CATEGORICAL ENCODING ==========")

# STEP 1: Label Encoding

label_encoder = LabelEncoder()

df['performance_label'] = label_encoder.fit_transform(df['performance'])

print("\n=== LABEL ENCODING ===")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print(df[['performance', 'performance_label']].head())

# STEP 2: One Hot Encoding

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

study_mode_encoded = onehot_encoder.fit_transform(df[['study_mode']])

onehot_df = pd.DataFrame(
    study_mode_encoded,
    columns=onehot_encoder.get_feature_names_out(['study_mode'])
)

print("\n=== ONE HOT ENCODING ===")
print(onehot_df.head())

# STEP 3: Ordinal Encoding

ordinal_encoder = OrdinalEncoder(
    categories=[['Low', 'Medium', 'High']]
)

df['performance_ordinal'] = ordinal_encoder.fit_transform(
    df[['performance']]
).astype(int)

print("\n=== ORDINAL ENCODING ===")
print(df[['performance', 'performance_ordinal']].head())

print("\nCategorical encoding completed!")
print("\nDataset saved successfully!")

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

print("\n========== FEATURE SCALING ==========")

numeric_features = [
    'age',
    'study_hrs',
    'midterm',
    'assignments',
    'attendance'
]

X = df[numeric_features]

# Standard Scaling
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(X)

# MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(X)

# Robust Scaling
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(X)

standard_df = pd.DataFrame(
    standard_scaled, columns=numeric_features
)

minmax_df = pd.DataFrame(
    minmax_scaled, columns=numeric_features
)

robust_df = pd.DataFrame(
    robust_scaled, columns=numeric_features
)

print("\n=== STANDARD SCALER ===")
print(standard_df.head().round(2))

print("\n=== MINMAX SCALER ===")
print(minmax_df.head().round(2))

print("\n=== ROBUST SCALER ===")
print(robust_df.head().round(2))

print("\nFeature scaling completed!")
# ================= FEATURE SELECTION =================

from sklearn.feature_selection import SelectKBest, f_regression

print("\n========== FEATURE SELECTION ==========")

# Create additional meaningful features
df['study_efficiency'] = (
    df['study_hrs'] * df['attendance'] / 100
)

df['score_gap'] = (
    df['assignments'] - df['midterm']
)

# Add One-Hot Encoded study mode
df['study_mode_Online'] = onehot_df['study_mode_Online']

# Select input features
feature_columns = [
    'age',
    'study_hrs',
    'midterm',
    'assignments',
    'attendance',
    'study_efficiency',
    'score_gap',
    'study_mode_Online'
]

X_features = df[feature_columns]

# Target variable
y = df['final_score']

# Select the top 5 features
selector = SelectKBest(
    score_func=f_regression,
    k=5
)

selector.fit(X_features, y)

# Display feature scores
feature_scores = pd.DataFrame({
    'Feature': feature_columns,
    'F_Score': selector.scores_,
    'P_Value': selector.pvalues_,
    'Selected': selector.get_support()
})

feature_scores = feature_scores.sort_values(
    by='F_Score',
    ascending=False
)

print("\n=== FEATURE SCORES ===")
print(feature_scores)

# Save results
feature_scores.to_csv(
    "outputs/feature_selection_results.csv",
    index=False
)

print("\nTop 5 feature selection completed!")
print("Results saved in outputs folder.")
# ================= SCALING DISTRIBUTION PLOTS =================

import os
import matplotlib.pyplot as plt

print("\n========== SCALING DISTRIBUTION PLOTS ==========")

os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(4, 2, figsize=(16, 20))

datasets = [
    ("Original", X),
    ("StandardScaler", standard_df),
    ("MinMaxScaler", minmax_df),
    ("RobustScaler", robust_df)
]

for row, (name, data) in enumerate(datasets):
    for col, feature in enumerate(["midterm", "study_hrs"]):
        axes[row, col].hist(
            data[feature],
            bins=12,
            edgecolor="black"
        )

        axes[row, col].set_title(f"{name} - {feature}")
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Frequency")

plt.tight_layout(pad=3.0, h_pad=3.0)

plt.savefig(
    "outputs/scaling_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Scaling comparison plot saved successfully!")
print("File location: outputs/scaling_comparison.png")