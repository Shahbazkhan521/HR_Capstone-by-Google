#!/usr/bin/env python3
"""
Improved HR Analysis with Enhanced Performance and Accuracy
- Advanced feature engineering
- Multiple model comparison
- Proper hyperparameter tuning
- Ensemble methods
- Performance optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Multiple models for comparison
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, VotingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Advanced models
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("XGBoost and/or LightGBM not available. Using only sklearn models.")

# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           roc_auc_score, precision_score, recall_score, f1_score,
                           roc_curve, precision_recall_curve)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.utils.class_weight import compute_class_weight

import joblib
import time
from datetime import datetime

class ImprovedHRAnalysis:
    """Enhanced HR Analysis with performance and accuracy improvements"""
    
    def __init__(self, data_path='HR_capstone_dataset.csv'):
        """Initialize with data loading and basic setup"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = None
        
        print(f"🚀 Initializing Improved HR Analysis")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_and_preprocess_data(self):
        """Enhanced data loading with better preprocessing"""
        print("\n📊 Loading and preprocessing data...")
        start_time = time.time()
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Turnover rate: {self.df['left'].mean():.2%}")
        
        # Handle duplicates more intelligently
        initial_size = len(self.df)
        self.df.drop_duplicates(inplace=True)
        duplicates_removed = initial_size - len(self.df)
        print(f"Removed {duplicates_removed} duplicates")
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values - handling them...")
            # Handle missing values if any
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        
        print(f"Data preprocessing completed in {time.time() - start_time:.2f}s")
        return self
    
    def advanced_feature_engineering(self):
        """Create advanced features for better model performance"""
        print("\n🔧 Performing advanced feature engineering...")
        start_time = time.time()
        
        # Create interaction features
        self.df['satisfaction_evaluation_ratio'] = (
            self.df['satisfaction_level'] / (self.df['last_evaluation'] + 1e-6)
        )
        
        # Workload intensity features
        self.df['hours_per_project'] = (
            self.df['average_montly_hours'] / self.df['number_project']
        )
        
        # Performance-satisfaction interaction
        self.df['high_performer_low_satisfaction'] = (
            (self.df['last_evaluation'] > 0.8) & (self.df['satisfaction_level'] < 0.4)
        ).astype(int)
        
        # Overwork indicator
        self.df['overworked'] = (
            (self.df['average_montly_hours'] > 250) & (self.df['number_project'] >= 6)
        ).astype(int)
        
        # Experience level based on time in company
        self.df['experience_level'] = pd.cut(
            self.df['time_spend_company'],
            bins=[0, 2, 4, 6, float('inf')],
            labels=['junior', 'mid', 'senior', 'expert']
        )
        
        # Workload category
        self.df['workload_category'] = pd.cut(
            self.df['average_montly_hours'],
            bins=[0, 160, 220, 280, float('inf')],
            labels=['low', 'medium', 'high', 'extreme']
        )
        
        # Performance category
        self.df['performance_category'] = pd.cut(
            self.df['last_evaluation'],
            bins=[0, 0.6, 0.8, 1.0],
            labels=['low_performer', 'average_performer', 'high_performer']
        )
        
        # Satisfaction category
        self.df['satisfaction_category'] = pd.cut(
            self.df['satisfaction_level'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['dissatisfied', 'neutral', 'satisfied']
        )
        
        # Create employee segment feature if not exists
        if 'employee_segment' not in self.df.columns:
            # Create segments based on satisfaction and performance
            conditions = [
                (self.df['satisfaction_level'] > 0.7) & (self.df['last_evaluation'] > 0.7),
                (self.df['satisfaction_level'] <= 0.4) & (self.df['last_evaluation'] > 0.7),
                (self.df['satisfaction_level'] > 0.7) & (self.df['last_evaluation'] <= 0.6),
                (self.df['satisfaction_level'] <= 0.4) & (self.df['last_evaluation'] <= 0.6)
            ]
            choices = ['star_employees', 'frustrated_employees', 'satisfied_low_performers', 'problem_employees']
            self.df['employee_segment'] = np.select(conditions, choices, default='average_employees')
        
        print(f"Feature engineering completed in {time.time() - start_time:.2f}s")
        print(f"Total features: {len(self.df.columns)}")
        return self
    
    def prepare_features_target(self):
        """Prepare features and target with advanced preprocessing"""
        print("\n⚙️ Preparing features and target variables...")
        start_time = time.time()
        
        # Define target
        y = self.df['left']
        
        # Define features (exclude target)
        X = self.df.drop(['left'], axis=1)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Preprocessing for numerical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform training data
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Get feature names for later use
        try:
            cat_feature_names = (self.preprocessor.named_transformers_['cat']
                               .named_steps['onehot']
                               .get_feature_names_out(categorical_features))
            self.feature_names = list(numeric_features) + list(cat_feature_names)
        except:
            self.feature_names = [f'feature_{i}' for i in range(self.X_train_processed.shape[1])]
        
        print(f"Features prepared in {time.time() - start_time:.2f}s")
        print(f"Training set shape: {self.X_train_processed.shape}")
        print(f"Test set shape: {self.X_test_processed.shape}")
        print(f"Class distribution - Train: {np.bincount(self.y_train)}")
        
        return self
    
    def setup_models(self):
        """Setup multiple models for comparison"""
        print("\n🤖 Setting up models for comparison...")
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        # Base models
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42
            ),
            'SVM': SVC(
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            'K-Neighbors': KNeighborsClassifier(
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB()
        }
        
        # Add advanced models if available
        if ADVANCED_MODELS_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            self.models['LightGBM'] = lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"Setup {len(self.models)} models for comparison")
        return self
    
    def evaluate_models(self, cv_folds=5):
        """Evaluate all models with cross-validation"""
        print(f"\n📈 Evaluating models with {cv_folds}-fold cross-validation...")
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Store results
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            start_time = time.time()
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, self.X_train_processed, self.y_train,
                    cv=skf, scoring='roc_auc', n_jobs=-1
                )
                
                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores,
                    'training_time': time.time() - start_time
                }
                
                print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}) "
                      f"[{time.time() - start_time:.2f}s]")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                model_scores[name] = {
                    'cv_mean': 0,
                    'cv_std': 0,
                    'cv_scores': [0],
                    'training_time': 0,
                    'error': str(e)
                }
        
        self.results['cv_scores'] = model_scores
        
        # Find best performing model
        best_model_name = max(model_scores.keys(), 
                            key=lambda x: model_scores[x]['cv_mean'])
        print(f"\n🏆 Best performing model: {best_model_name} "
              f"(CV AUC: {model_scores[best_model_name]['cv_mean']:.4f})")
        
        return model_scores
    
    def hyperparameter_tuning(self, model_name=None):
        """Perform hyperparameter tuning for the best model"""
        if model_name is None:
            model_name = max(self.results['cv_scores'].keys(),
                           key=lambda x: self.results['cv_scores'][x]['cv_mean'])
        
        print(f"\n🔍 Performing hyperparameter tuning for {model_name}...")
        start_time = time.time()
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name in param_grids:
            # Use RandomizedSearchCV for efficiency
            random_search = RandomizedSearchCV(
                estimator=self.models[model_name],
                param_distributions=param_grids[model_name],
                n_iter=50,  # Number of parameter combinations to try
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(self.X_train_processed, self.y_train)
            
            self.best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {best_score:.4f}")
            print(f"Hyperparameter tuning completed in {time.time() - start_time:.2f}s")
            
            return best_params, best_score
        else:
            print(f"No parameter grid defined for {model_name}, using default parameters")
            self.best_model = self.models[model_name]
            self.best_model.fit(self.X_train_processed, self.y_train)
            return {}, 0
    
    def create_ensemble_model(self):
        """Create ensemble model combining top performers"""
        print("\n🎭 Creating ensemble model...")
        start_time = time.time()
        
        # Select top 3 models based on CV scores
        sorted_models = sorted(
            self.results['cv_scores'].items(),
            key=lambda x: x[1]['cv_mean'],
            reverse=True
        )
        
        top_models = sorted_models[:3]
        print("Top 3 models for ensemble:")
        for name, scores in top_models:
            print(f"  {name}: {scores['cv_mean']:.4f}")
        
        # Create voting classifier
        voting_models = [(name, self.models[name]) for name, _ in top_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=voting_models,
            voting='soft',  # Use predicted probabilities
            n_jobs=-1
        )
        
        # Train ensemble
        self.ensemble_model.fit(self.X_train_processed, self.y_train)
        
        print(f"Ensemble model created in {time.time() - start_time:.2f}s")
        return self
    
    def final_evaluation(self):
        """Perform final evaluation on test set"""
        print("\n📊 Final evaluation on test set...")
        
        models_to_evaluate = {}
        
        # Add best individual model
        if self.best_model is not None:
            models_to_evaluate['Best Individual Model'] = self.best_model
        
        # Add ensemble model
        if hasattr(self, 'ensemble_model'):
            models_to_evaluate['Ensemble Model'] = self.ensemble_model
        
        final_results = {}
        
        for name, model in models_to_evaluate.items():
            print(f"\nEvaluating {name}:")
            
            # Predictions
            y_pred = model.predict(self.X_test_processed)
            y_pred_proba = model.predict_proba(self.X_test_processed)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            final_results[name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print metrics
            for metric_name, value in metrics.items():
                print(f"  {metric_name.capitalize()}: {value:.4f}")
            
            print(f"\nClassification Report for {name}:")
            print(classification_report(self.y_test, y_pred))
        
        self.results['final_evaluation'] = final_results
        return final_results
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n🔍 Analyzing feature importance...")
        
        if self.best_model is None:
            print("No best model available for feature importance analysis")
            return
        
        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            # Create importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(15), y='feature', x='importance')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return feature_importance
        else:
            print("Model does not support feature importance analysis")
            return None
    
    def save_models(self):
        """Save the best models"""
        print("\n💾 Saving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best individual model
        if self.best_model is not None:
            filename = f'best_model_{timestamp}.pkl'
            joblib.dump(self.best_model, filename)
            print(f"Best individual model saved as: {filename}")
        
        # Save ensemble model
        if hasattr(self, 'ensemble_model'):
            filename = f'ensemble_model_{timestamp}.pkl'
            joblib.dump(self.ensemble_model, filename)
            print(f"Ensemble model saved as: {filename}")
        
        # Save preprocessor
        filename = f'preprocessor_{timestamp}.pkl'
        joblib.dump(self.preprocessor, filename)
        print(f"Preprocessor saved as: {filename}")
    
    def run_complete_analysis(self):
        """Run the complete improved analysis"""
        print("=" * 70)
        print("🚀 STARTING IMPROVED HR ANALYSIS")
        print("=" * 70)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Advanced feature engineering
            self.advanced_feature_engineering()
            
            # Step 3: Prepare features and target
            self.prepare_features_target()
            
            # Step 4: Setup models
            self.setup_models()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Hyperparameter tuning
            self.hyperparameter_tuning()
            
            # Step 7: Create ensemble model
            self.create_ensemble_model()
            
            # Step 8: Final evaluation
            self.final_evaluation()
            
            # Step 9: Feature importance analysis
            self.feature_importance_analysis()
            
            # Step 10: Save models
            self.save_models()
            
            total_time = time.time() - total_start_time
            print(f"\n🎉 Complete analysis finished in {total_time:.2f}s")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the improved analysis"""
    
    # Create and run analysis
    analysis = ImprovedHRAnalysis()
    results = analysis.run_complete_analysis()
    
    if results:
        print("\n" + "="*70)
        print("📈 ANALYSIS SUMMARY")
        print("="*70)
        
        # Print final results summary
        if 'final_evaluation' in results:
            for model_name, model_results in results['final_evaluation'].items():
                print(f"\n{model_name}:")
                for metric, value in model_results['metrics'].items():
                    print(f"  {metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()