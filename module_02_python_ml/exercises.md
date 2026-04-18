# Module 2 — Exercises

## Conceptual Questions

### Exercise 2.1: Broadcasting
What is the resulting shape of the following operations?
1. `np.zeros((3, 4)) + np.ones((4,))`
2. `np.zeros((3, 1)) + np.ones((1, 4))`
3. `np.zeros((3, 4)) + np.ones((3,))` — Will this work? Why or why not?

### Exercise 2.2: Data Leakage
Explain what's wrong with this pipeline:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale ALL data
X_train, X_test = train_test_split(X_scaled)  # Then split
```
How should it be done instead?

### Exercise 2.3: Missing Data Strategy
For each scenario, which missing data strategy would you use and why?
1. A medical dataset where 2% of blood pressure readings are missing
2. A time-series dataset (daily temperature) with 15% gaps
3. A feature with 80% missing values
4. A survey where respondents skip sensitive questions (e.g., income)

---

## Implementation Exercises

### Exercise 2.4: NumPy Challenge
Using only NumPy (no loops!), implement:
1. A function that normalizes each row of a matrix to sum to 1 (like softmax without exp)
2. A function that computes the cosine similarity matrix for a set of vectors
3. A function that finds the k nearest neighbors of each point in a dataset

### Exercise 2.5: Pandas Exploration
Using the Iris dataset (`sklearn.datasets.load_iris`):
1. Load it into a Pandas DataFrame
2. Add species names using the target mapping
3. Compute the mean and std of each feature per species
4. Find which feature has the highest variance ratio between species
5. Create a crosstab of species vs petal length bins

### Exercise 2.6: Visualization Gallery
Create a comprehensive visualization dashboard for the Wine dataset:
1. Distribution of each class
2. Box plots of all 13 features grouped by class
3. Pair plot of the top 4 most discriminative features
4. Radar/spider chart comparing the mean feature values for each class

### Exercise 2.7: Custom Pipeline
Build a data pipeline function that:
1. Accepts a CSV file path
2. Automatically detects numeric vs categorical columns
3. Imputes numeric columns with median, categorical with mode
4. One-hot encodes categorical variables
5. Scales numeric features
6. Returns train/val/test splits
Test with any public CSV dataset.

---

## Challenge Exercise

### Exercise 2.8: Automated EDA Report
Write a function `auto_eda(df, target_col)` that:
1. Generates a comprehensive HTML or markdown report
2. Includes: shape, dtypes, missing values, distributions, correlations
3. If a target column is specified, shows feature importance
4. Generates all relevant plots and saves them
5. Provides data quality recommendations

This mimics tools like `pandas-profiling` but built from scratch!
