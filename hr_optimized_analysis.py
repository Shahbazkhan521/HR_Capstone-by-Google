#!/usr/bin/env python3
"""
Optimized HR Analysis with Enhanced Performance and Accuracy
- Faster execution while maintaining accuracy improvements
- Essential feature engineering and model optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import RandomizedSearchCV
import joblib
import time
from datetime import datetime

class OptimizedHRAnalysis:
    """Optimized HR Analysis focusing on key improvements"""
    
    def __init__(self, data_path='HR_capstone_dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.best_model = None
        
        print(f"🚀 Optimized HR Analysis Started")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_and_engineer_features(self):
        """Combined data loading and feature engineering for efficiency"""
        print("\n📊 Loading data and engineering features...")
        start_time = time.time()
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Initial dataset shape: {self.df.shape}")
        
        # Remove duplicates
        initial_size = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_size - len(self.df)} duplicates")
        
        # Create essential engineered features
        self.df['satisfaction_evaluation_ratio'] = (
            self.df['satisfaction_level'] / (self.df['last_evaluation'] + 1e-6)
        )
        
        self.df['hours_per_project'] = (
            self.df['average_montly_hours'] / self.df['number_project']
        )
        
        self.df['high_performer_low_satisfaction'] = (
            (self.df['last_evaluation'] > 0.8) & (self.df['satisfaction_level'] < 0.4)
        ).astype(int)
        
        self.df['overworked'] = (
            (self.df['average_montly_hours'] > 250) & (self.df['number_project'] >= 6)
        ).astype(int)
        
        # Create employee segment if not exists
        if 'employee_segment' not in self.df.columns:
            conditions = [
                (self.df['satisfaction_level'] > 0.7) & (self.df['last_evaluation'] > 0.7),
                (self.df['satisfaction_level'] <= 0.4) & (self.df['last_evaluation'] > 0.7),
                (self.df['satisfaction_level'] > 0.7) & (self.df['last_evaluation'] <= 0.6),
                (self.df['satisfaction_level'] <= 0.4) & (self.df['last_evaluation'] <= 0.6)
            ]
            choices = ['stars', 'frustrated', 'satisfied_low', 'problems']
            self.df['employee_segment'] = np.select(conditions, choices, default='average')
        
        print(f"Feature engineering completed in {time.time() - start_time:.2f}s")
        print(f"Final features: {self.df.columns.tolist()}")
        return self
    
    def prepare_data(self):
        """Prepare data for modeling"""
        print("\n⚙️ Preparing data for modeling...")
        start_time = time.time()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['Department', 'salary', 'employee_segment']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le
        
        # Define features and target
        X = self.df.drop(['left'], axis=1)
        y = self.df['left']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        self.scaler = scaler
        self.label_encoders = label_encoders
        
        print(f"Data preparation completed in {time.time() - start_time:.2f}s")
        print(f"Training shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        print(f"Class distribution: {np.bincount(self.y_train)}")
        
        return self
    
    def setup_and_evaluate_models(self):
        """Setup and evaluate key models efficiently"""
        print("\n🤖 Setting up and evaluating models...")
        start_time = time.time()
        
        # Define models with optimized parameters
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Evaluate models with cross-validation
        cv_results = {}
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            model_start = time.time()
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=3, scoring='roc_auc', n_jobs=-1
            )
            
            mean_score = cv_scores.mean()
            cv_results[name] = {
                'cv_mean': mean_score,
                'cv_std': cv_scores.std(),
                'time': time.time() - model_start
            }
            
            print(f"  CV AUC: {mean_score:.4f} (+/- {cv_scores.std()*2:.4f}) "
                  f"[{time.time() - model_start:.2f}s]")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
        
        self.results['cv_results'] = cv_results
        print(f"\n🏆 Best model: {best_model_name} (AUC: {best_score:.4f})")
        print(f"Model evaluation completed in {time.time() - start_time:.2f}s")
        
        return best_model_name
    
    def optimize_best_model(self, model_name):
        """Optimize the best performing model"""
        print(f"\n🔍 Optimizing {model_name}...")
        start_time = time.time()
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        }
        
        if model_name in param_grids:
            # Use RandomizedSearchCV for efficiency
            random_search = RandomizedSearchCV(
                estimator=self.models[model_name],
                param_distributions=param_grids[model_name],
                n_iter=20,  # Limited iterations for speed
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            
            random_search.fit(self.X_train, self.y_train)
            
            self.best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {best_score:.4f}")
        else:
            # For models without parameter grid, use default
            self.best_model = self.models[model_name]
            self.best_model.fit(self.X_train, self.y_train)
            best_params = "Default parameters"
            best_score = self.results['cv_results'][model_name]['cv_mean']
        
        print(f"Model optimization completed in {time.time() - start_time:.2f}s")
        return best_params, best_score
    
    def create_ensemble(self):
        """Create an ensemble of top models"""
        print("\n🎭 Creating ensemble model...")
        start_time = time.time()
        
        # Select top 3 models
        sorted_models = sorted(
            self.results['cv_results'].items(),
            key=lambda x: x[1]['cv_mean'],
            reverse=True
        )[:3]
        
        voting_models = [(name, self.models[name]) for name, _ in sorted_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            n_jobs=-1
        )
        
        self.ensemble_model.fit(self.X_train, self.y_train)
        
        print(f"Ensemble created with: {[name for name, _ in voting_models]}")
        print(f"Ensemble creation completed in {time.time() - start_time:.2f}s")
        return self
    
    def final_evaluation(self):
        """Evaluate final models on test set"""
        print("\n📊 Final evaluation on test set...")
        
        models_to_test = {
            'Best Individual Model': self.best_model,
            'Ensemble Model': self.ensemble_model
        }
        
        final_results = {}
        
        for name, model in models_to_test.items():
            print(f"\n{name} Results:")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            final_results[name] = metrics
            
            # Print metrics
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
        
        self.results['final_results'] = final_results
        return final_results
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print("\n🔍 Analyzing feature importance...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(15)
            sns.barplot(data=top_features, y='feature', x='importance')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return feature_importance
        else:
            print("Feature importance not available for this model type")
            return None
    
    def performance_comparison(self):
        """Compare performance improvements"""
        print("\n📈 Performance Comparison Summary:")
        print("="*50)
        
        if 'final_results' in self.results:
            for model_name, metrics in self.results['final_results'].items():
                print(f"\n{model_name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")
                print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Print key improvements
        print(f"\n🎯 Key Improvements Implemented:")
        print(f"  ✓ Advanced feature engineering (4 new features)")
        print(f"  ✓ Multiple model comparison")
        print(f"  ✓ Hyperparameter optimization")
        print(f"  ✓ Ensemble method")
        print(f"  ✓ Proper cross-validation")
        print(f"  ✓ Class imbalance handling")
        
    def save_models(self):
        """Save optimized models"""
        print("\n💾 Saving optimized models...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, f'optimized_best_model_{timestamp}.pkl')
            print(f"Best model saved")
        
        # Save ensemble
        if hasattr(self, 'ensemble_model'):
            joblib.dump(self.ensemble_model, f'optimized_ensemble_{timestamp}.pkl')
            print(f"Ensemble model saved")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f'scaler_{timestamp}.pkl')
        joblib.dump(self.label_encoders, f'label_encoders_{timestamp}.pkl')
        print(f"Preprocessing components saved")
    
    def run_complete_analysis(self):
        """Run the complete optimized analysis"""
        print("="*70)
        print("🚀 OPTIMIZED HR ANALYSIS")
        print("="*70)
        
        total_start = time.time()
        
        try:
            # Step 1: Load data and engineer features
            self.load_and_engineer_features()
            
            # Step 2: Prepare data
            self.prepare_data()
            
            # Step 3: Setup and evaluate models
            best_model_name = self.setup_and_evaluate_models()
            
            # Step 4: Optimize best model
            self.optimize_best_model(best_model_name)
            
            # Step 5: Create ensemble
            self.create_ensemble()
            
            # Step 6: Final evaluation
            self.final_evaluation()
            
            # Step 7: Feature importance analysis
            self.analyze_feature_importance()
            
            # Step 8: Performance comparison
            self.performance_comparison()
            
            # Step 9: Save models
            self.save_models()
            
            total_time = time.time() - total_start
            print(f"\n🎉 Analysis completed successfully in {total_time:.2f}s")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # Run optimized analysis
    analysis = OptimizedHRAnalysis()
    results = analysis.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()