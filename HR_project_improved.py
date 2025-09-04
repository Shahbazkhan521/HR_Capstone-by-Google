#!/usr/bin/env python3
"""
HR Project - Improved Version
This script converts the original notebook improvements into an executable Python script
with enhanced performance and accuracy features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           roc_auc_score, precision_score, recall_score, f1_score,
                           roc_curve, auc)
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """Main execution function"""
    print("🚀 HR CAPSTONE PROJECT - IMPROVED VERSION")
    print("="*60)
    
    # 1. Data Loading and Basic Info
    print("\n📊 STEP 1: Data Loading and Basic Information")
    print("-" * 40)
    
    df = pd.read_csv('HR_capstone_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nFirst few rows:")
    print(df.head())
    
    # 2. Data Quality Assessment
    print("\n🔍 STEP 2: Data Quality Assessment")
    print("-" * 40)
    
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    initial_size = len(df)
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates found: {duplicates}")
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_size - len(df)} duplicates")
    print(f"Final dataset shape: {df.shape}")
    
    # 3. Exploratory Data Analysis
    print("\n📈 STEP 3: Exploratory Data Analysis")
    print("-" * 40)
    
    # Turnover rate
    turnover_rate = df['left'].mean()
    print(f"Overall turnover rate: {turnover_rate:.2%}")
    
    # Class distribution
    print(f"Class distribution:")
    print(df['left'].value_counts())
    print(df['left'].value_counts(normalize=True))
    
    # Basic statistics
    print(f"\nNumerical features statistics:")
    print(df.describe())
    
    # Create basic visualizations
    create_basic_visualizations(df)
    
    # 4. Advanced Feature Engineering
    print("\n🔧 STEP 4: Advanced Feature Engineering")
    print("-" * 40)
    
    df = perform_feature_engineering(df)
    print(f"Features after engineering: {df.columns.tolist()}")
    
    # 5. Data Preprocessing
    print("\n⚙️ STEP 5: Data Preprocessing")
    print("-" * 40)
    
    X, y, scaler, label_encoders = preprocess_data(df)
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 6. Model Training and Evaluation
    print("\n🤖 STEP 6: Model Training and Evaluation")
    print("-" * 40)
    
    results = train_and_evaluate_models(X, y)
    
    # 7. Final Model Performance
    print("\n📊 STEP 7: Final Model Performance Report")
    print("-" * 40)
    
    generate_final_report(results)
    
    # 8. Feature Importance Analysis
    print("\n🔍 STEP 8: Feature Importance Analysis")
    print("-" * 40)
    
    analyze_feature_importance(results['best_model'], X.columns)
    
    print("\n🎉 Analysis completed successfully!")
    print("Check generated visualizations and saved models.")

def create_basic_visualizations(df):
    """Create basic visualizations"""
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Turnover distribution
    df['left'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Turnover Distribution')
    axes[0,0].set_xlabel('Left (0=Stayed, 1=Left)')
    axes[0,0].set_ylabel('Count')
    
    # Satisfaction level distribution by turnover
    df.boxplot(column='satisfaction_level', by='left', ax=axes[0,1])
    axes[0,1].set_title('Satisfaction Level by Turnover')
    
    # Average monthly hours by turnover
    df.boxplot(column='average_montly_hours', by='left', ax=axes[0,2])
    axes[0,2].set_title('Monthly Hours by Turnover')
    
    # Department wise turnover
    dept_turnover = df.groupby('Department')['left'].mean().sort_values(ascending=False)
    dept_turnover.plot(kind='bar', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('Turnover Rate by Department')
    axes[1,0].set_xlabel('Department')
    axes[1,0].set_ylabel('Turnover Rate')
    
    # Salary wise turnover
    salary_turnover = df.groupby('salary')['left'].mean()
    salary_turnover.plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Turnover Rate by Salary Level')
    axes[1,1].set_xlabel('Salary Level')
    axes[1,1].set_ylabel('Turnover Rate')
    
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
    axes[1,2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Basic visualizations saved as 'eda_visualizations.png'")

def perform_feature_engineering(df):
    """Perform advanced feature engineering"""
    
    # Interaction features
    df['satisfaction_evaluation_ratio'] = df['satisfaction_level'] / (df['last_evaluation'] + 1e-6)
    df['hours_per_project'] = df['average_montly_hours'] / df['number_project']
    
    # Risk indicators
    df['high_performer_low_satisfaction'] = (
        (df['last_evaluation'] > 0.8) & (df['satisfaction_level'] < 0.4)
    ).astype(int)
    
    df['overworked'] = (
        (df['average_montly_hours'] > 250) & (df['number_project'] >= 6)
    ).astype(int)
    
    # Enhanced employee segmentation
    if 'employee_segment' not in df.columns:
        conditions = [
            (df['satisfaction_level'] > 0.7) & (df['last_evaluation'] > 0.7),
            (df['satisfaction_level'] <= 0.4) & (df['last_evaluation'] > 0.7),
            (df['satisfaction_level'] > 0.7) & (df['last_evaluation'] <= 0.6),
            (df['satisfaction_level'] <= 0.4) & (df['last_evaluation'] <= 0.6)
        ]
        choices = ['star_employees', 'frustrated_employees', 'satisfied_low_performers', 'problem_employees']
        df['employee_segment'] = np.select(conditions, choices, default='average_employees')
    
    print(f"Added {len(['satisfaction_evaluation_ratio', 'hours_per_project', 'high_performer_low_satisfaction', 'overworked'])} new features")
    
    return df

def preprocess_data(df):
    """Preprocess data for modeling"""
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Department', 'salary', 'employee_segment']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Define features and target
    X = df.drop(['left'], axis=1)
    y = df['left']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y, scaler, label_encoders

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
    }
    
    # Evaluate models
    model_results = {}
    best_score = 0
    best_model = None
    best_name = None
    
    print("Model evaluation results:")
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Train model
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        model_results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name:20} - CV AUC: {metrics['cv_mean']:.4f} (+/-{metrics['cv_std']*2:.4f})")
        print(f"{'':20}   Test AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['cv_mean'] > best_score:
            best_score = metrics['cv_mean']
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (CV AUC: {best_score:.4f})")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    # Ensemble predictions
    ensemble_pred = ensemble.predict(X_test)
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred),
        'recall': recall_score(y_test, ensemble_pred),
        'f1': f1_score(y_test, ensemble_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_proba)
    }
    
    model_results['Ensemble'] = {
        'model': ensemble,
        'metrics': ensemble_metrics,
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba
    }
    
    print(f"Ensemble Model      - Test AUC: {ensemble_metrics['roc_auc']:.4f}, "
          f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
    
    # Store test data for later use
    model_results['test_data'] = (X_test, y_test)
    model_results['best_model'] = best_model
    model_results['best_name'] = best_name
    
    return model_results

def generate_final_report(results):
    """Generate final performance report"""
    
    print("\nFINAL MODEL PERFORMANCE REPORT:")
    print("=" * 50)
    
    # Model comparison table
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'ROC AUC':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        if name not in ['test_data', 'best_model', 'best_name']:
            metrics = result['metrics']
            print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['roc_auc']:<8.4f}")
    
    # Detailed classification report for best model
    X_test, y_test = results['test_data']
    best_name = results['best_name']
    best_predictions = results[best_name]['predictions']
    
    print(f"\nDetailed Classification Report - {best_name}:")
    print(classification_report(y_test, best_predictions))
    
    print(f"\nConfusion Matrix - {best_name}:")
    print(confusion_matrix(y_test, best_predictions))
    
    # Save models
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    joblib.dump(results['best_model'], f'hr_best_model_{timestamp}.pkl')
    joblib.dump(results['Ensemble']['model'], f'hr_ensemble_model_{timestamp}.pkl')
    
    print(f"\nModels saved with timestamp: {timestamp}")

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('Top 15 Feature Importances (Improved Model)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance_final.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance plot saved as 'feature_importance_final.png'")
        
        return importance_df
    else:
        print("Feature importance not available for this model type")
        return None

if __name__ == "__main__":
    main()