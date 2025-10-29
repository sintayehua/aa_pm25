import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.settings import MODEL_CONFIG, MODELS_DIR

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=MODEL_CONFIG['random_state']),
            'xgboost': xgb.XGBRegressor(random_state=MODEL_CONFIG['random_state']),
            'lightgbm': lgb.LGBMRegressor(random_state=MODEL_CONFIG['random_state']),
            'gradient_boosting': GradientBoostingRegressor(random_state=MODEL_CONFIG['random_state']),
            'ridge': Ridge(random_state=MODEL_CONFIG['random_state']),
            'linear': LinearRegression()
        }
        
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
        }
        
        self.best_model = None
        self.best_model_name = None
        self.cv_scores = {}
        self.feature_importance = None
        
    def train_models(self, X, y, preprocessor):
        """Train multiple models and select the best one"""
        print("=== MODEL TRAINING ===")
        
        # Preprocess features
        X_processed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'],
            shuffle=False  # Important for time series
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Hyperparameter tuning for complex models
                if name in self.param_grids:
                    grid_search = GridSearchCV(
                        model, self.param_grids[name],
                        cv=tscv, scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"  Best params: {grid_search.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    best_model, X_train, y_train,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                cv_rmse = np.sqrt(-cv_scores)
                
                # Predictions
                y_pred = best_model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'model': best_model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std(),
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
                print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        # Select best model based on RMSE
        if results:
            self.best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
            self.best_model = results[self.best_model_name]['model']
            self.cv_scores = results
            
            print(f"\n=== BEST MODEL: {self.best_model_name.upper()} ===")
            best_result = results[self.best_model_name]
            print(f"MAE: {best_result['mae']:.2f}")
            print(f"RMSE: {best_result['rmse']:.2f}")
            print(f"R²: {best_result['r2']:.2f}")
            
            # Feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nTop 10 most important features:")
                print(self.feature_importance.head(10))
        
        return results, (X_test, y_test), feature_names
    
    def predict(self, X, preprocessor):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        X_processed = preprocessor.transform(X)
        return self.best_model.predict(X_processed)
    
    def save_model(self, preprocessor, feature_names, filename='pm25_model'):
        """Save the trained model and preprocessor"""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.best_model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'model_name': self.best_model_name,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance
        }
        
        filepath = os.path.join(MODELS_DIR, f'{filename}.joblib')
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename='pm25_model'):
        """Load a trained model"""
        filepath = os.path.join(MODELS_DIR, f'{filename}.joblib')
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.cv_scores = model_data['cv_scores']
        self.feature_importance = model_data['feature_importance']
        
        print(f"Model loaded from {filepath}")
        return model_data['preprocessor'], model_data['feature_names']