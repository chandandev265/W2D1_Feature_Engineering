# W2D1 — Feature Engineering & Encoding

## Internship Project
Cynaris Solutions — SOC Analyst Internship

## Project Overview
This project demonstrates data preprocessing and feature engineering using Python.

The tasks include:
- Dataset creation and inspection
- Categorical data encoding
- Feature scaling
- Feature selection using SelectKBest
- Distribution visualization
- Simulated malware alert triage documentation

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
W2D1_Feature_Engineering/
|
|-- data/
|   |-- student_dataset.csv
|
|-- outputs/
|   |-- feature_selection_results.csv
|   |-- scaling_comparison.png
|
|-- feature_engineering.py
|-- malware_triage_log.md
|-- requirements.txt
|-- README.md
|-- self_review.md
```

## How to Run

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Python script:

   ```bash
   python feature_engineering.py
   ```

## Feature Engineering Tasks

### Categorical Encoding
- LabelEncoder
- OneHotEncoder
- OrdinalEncoder

### Feature Scaling
- StandardScaler
- MinMaxScaler
- RobustScaler

### Feature Selection
SelectKBest with f_regression was used to select the top five features associated with the final score.

### Visualization
The scaling comparison plot is saved at:

`outputs/scaling_comparison.png`

### Malware Alert Triage
A separate Markdown log documents five simulated malware execution scenarios.

**Note:** The dataset and malware scenarios are simulated training materials. They are not actual student records or verified endpoint incidents.