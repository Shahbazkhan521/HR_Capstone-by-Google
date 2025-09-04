#!/usr/bin/env python3
"""
Create a sample HR dataset for testing performance improvements
Based on the structure observed in the notebook
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic HR dataset with realistic characteristics
n_samples = 15000

# Generate base features using make_classification
X, y = make_classification(
    n_samples=n_samples,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_clusters_per_class=1,
    weights=[0.76, 0.24],  # Imbalanced classes (24% turnover rate)
    random_state=42
)

# Create DataFrame with realistic HR features
df = pd.DataFrame()

# Satisfaction level (0-1)
df['satisfaction_level'] = np.clip(X[:, 0] * 0.2 + 0.5, 0.09, 1.0)

# Last evaluation (0.36-1.0)
df['last_evaluation'] = np.clip(X[:, 1] * 0.15 + 0.7, 0.36, 1.0)

# Number of projects (2-7)
df['number_project'] = np.clip(np.round(X[:, 2] * 1.2 + 4), 2, 7).astype(int)

# Average monthly hours (96-310)
df['average_montly_hours'] = np.clip(np.round(X[:, 3] * 40 + 200), 96, 310).astype(int)

# Time spent in company (2-10 years)
df['time_spend_company'] = np.clip(np.round(X[:, 4] * 1.5 + 4), 2, 10).astype(int)

# Work accident (0 or 1)
df['Work_accident'] = (X[:, 5] > 0.3).astype(int)

# Target variable
df['left'] = y

# Promotion in last 5 years (0 or 1) - rare event
df['promotion_last_5years'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])

# Department (categorical)
departments = ['sales', 'technical', 'support', 'IT', 'product_mng', 
               'marketing', 'RandD', 'accounting', 'hr', 'management']
df['Department'] = np.random.choice(departments, n_samples, 
                                  p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.015, 0.005])

# Salary (categorical)
df['salary'] = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.5, 0.4, 0.1])

# Add some logical relationships to make it more realistic
# Lower satisfaction should correlate with leaving
mask_low_satisfaction = df['satisfaction_level'] < 0.3
df.loc[mask_low_satisfaction, 'left'] = np.random.choice([0, 1], 
                                                        sum(mask_low_satisfaction), 
                                                        p=[0.2, 0.8])

# High performers with low satisfaction are flight risks
mask_high_perf_low_sat = (df['last_evaluation'] > 0.8) & (df['satisfaction_level'] < 0.4)
df.loc[mask_high_perf_low_sat, 'left'] = 1

# People with many projects and high hours might leave
mask_overworked = (df['number_project'] >= 6) & (df['average_montly_hours'] > 280)
df.loc[mask_overworked, 'left'] = np.random.choice([0, 1], sum(mask_overworked), p=[0.3, 0.7])

# Add duplicates to simulate the original dataset
n_duplicates = 3008
duplicate_indices = np.random.choice(df.index, n_duplicates, replace=True)
df_duplicates = df.iloc[duplicate_indices].copy()
df = pd.concat([df, df_duplicates], ignore_index=True)

print(f"Created dataset with {len(df)} rows")
print(f"Turnover rate: {df['left'].mean():.2%}")
print(f"Duplicates: {df.duplicated().sum()}")
print("\nDataset info:")
print(df.info())
print("\nSample data:")
print(df.head())

# Save the dataset
df.to_csv('HR_capstone_dataset.csv', index=False)
print("\nDataset saved as 'HR_capstone_dataset.csv'")