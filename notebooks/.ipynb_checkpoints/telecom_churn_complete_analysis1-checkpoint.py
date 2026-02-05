"""
Comprehensive Telecom Customer Churn Prediction Pipeline
=========================================================
This script provides end-to-end ML pipeline including:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Selection & Training
4. Hyperparameter Tuning
5. Model Evaluation
6. Model Explainability
7. Prediction Inference
8. User Testing Interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
from datetime import datetime

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class TelecomChurnPredictor:
    """Complete ML pipeline for customer churn prediction"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, df=None):
        """Load data from file or DataFrame"""
        if df is not None:
            self.df = df.copy()
        elif self.data_path:
            self.df = pd.read_csv(self.data_path)
        else:
            # Create sample data based on provided example
            print("Creating sample dataset...")
            self.df = self._create_sample_data()
        
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'customerID': [f'CUST-{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples).round(2),
            'TotalCharges': np.random.uniform(18.0, 8500.0, n_samples).round(2),
        }
        
        df = pd.DataFrame(data)
        
        # Create churn based on realistic patterns
        churn_prob = (
            (df['Contract'] == 'Month-to-month') * 0.3 +
            (df['tenure'] < 12) * 0.25 +
            (df['MonthlyCharges'] > 80) * 0.2 +
            (df['TechSupport'] == 'No') * 0.15 +
            np.random.random(n_samples) * 0.1
        )
        df['Churn'] = (churn_prob > 0.5).astype(int)
        df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        
        return df
    
    def exploratory_data_analysis(self, save_plots=True):
        """Comprehensive EDA"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic info
        print("\n1. Dataset Overview:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n2. Data Types:")
        print(self.df.dtypes.value_counts())
        
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   No missing values!")
        
        print("\n4. Duplicate Rows:")
        print(f"   {self.df.duplicated().sum()} duplicates found")
        
        print("\n5. Target Variable Distribution:")
        churn_dist = self.df['Churn'].value_counts()
        print(churn_dist)
        print(f"\n   Churn Rate: {(churn_dist['Yes'] / len(self.df)) * 100:.2f}%")
        
        print("\n6. Numerical Features Statistics:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        print("\n7. Categorical Features:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['customerID', 'Churn']:
                print(f"\n   {col}: {self.df[col].nunique()} unique values")
                print(self.df[col].value_counts().head())
        
        if save_plots:
            self._create_eda_plots()
        
        return self._generate_eda_insights()
    
    def _create_eda_plots(self):
        """Create comprehensive EDA visualizations"""
        print("\nGenerating EDA visualizations...")
        
        # 1. Target distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
                       colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0, 0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
        
        # Churn by Contract Type
        pd.crosstab(self.df['Contract'], self.df['Churn'], normalize='index').plot(
            kind='bar', stacked=False, ax=axes[0, 1], color=['#2ecc71', '#e74c3c']
        )
        axes[0, 1].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Contract Type')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].legend(title='Churn')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Tenure distribution
        self.df[self.df['Churn'] == 'No']['tenure'].hist(bins=30, alpha=0.5, 
                                                          label='No Churn', ax=axes[1, 0], color='#2ecc71')
        self.df[self.df['Churn'] == 'Yes']['tenure'].hist(bins=30, alpha=0.5, 
                                                           label='Churn', ax=axes[1, 0], color='#e74c3c')
        axes[1, 0].set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Tenure (months)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Monthly charges
        self.df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1, 1])
        axes[1, 1].set_title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Churn')
        axes[1, 1].set_ylabel('Monthly Charges ($)')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('/home/claude/eda_overview.png', dpi=300, bbox_inches='tight')
        print("   Saved: eda_overview.png")
        plt.close()
        
        # 2. Correlation heatmap
        df_encoded = self.df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col != 'customerID':
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
        
        plt.figure(figsize=(14, 10))
        correlation = df_encoded.drop('customerID', axis=1, errors='ignore').corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("   Saved: correlation_heatmap.png")
        plt.close()
        
        # 3. Feature importance plot (preliminary)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn by Internet Service
        pd.crosstab(self.df['InternetService'], self.df['Churn'], normalize='index').plot(
            kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c']
        )
        axes[0, 0].set_title('Churn Rate by Internet Service', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Internet Service')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Churn by Payment Method
        pd.crosstab(self.df['PaymentMethod'], self.df['Churn'], normalize='index').plot(
            kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c']
        )
        axes[0, 1].set_title('Churn Rate by Payment Method', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Payment Method')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Churn by Senior Citizen
        pd.crosstab(self.df['SeniorCitizen'], self.df['Churn'], normalize='index').plot(
            kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c']
        )
        axes[1, 0].set_title('Churn Rate by Senior Citizen Status', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Senior Citizen (0=No, 1=Yes)')
        axes[1, 0].set_ylabel('Proportion')
        
        # Churn by Tech Support
        pd.crosstab(self.df['TechSupport'], self.df['Churn'], normalize='index').plot(
            kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c']
        )
        axes[1, 1].set_title('Churn Rate by Tech Support', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Tech Support')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/claude/categorical_analysis.png', dpi=300, bbox_inches='tight')
        print("   Saved: categorical_analysis.png")
        plt.close()
    
    def _generate_eda_insights(self):
        """Generate key insights from EDA"""
        insights = {
            'total_customers': len(self.df),
            'churn_rate': (self.df['Churn'] == 'Yes').sum() / len(self.df) * 100,
            'avg_tenure': self.df['tenure'].mean(),
            'avg_monthly_charges': self.df['MonthlyCharges'].mean(),
            'contract_churn': {},
            'senior_churn_rate': None
        }
        
        # Churn by contract type
        for contract in self.df['Contract'].unique():
            mask = self.df['Contract'] == contract
            churn_rate = (self.df[mask]['Churn'] == 'Yes').sum() / mask.sum() * 100
            insights['contract_churn'][contract] = round(churn_rate, 2)
        
        # Senior citizen churn
        if 'SeniorCitizen' in self.df.columns:
            senior_mask = self.df['SeniorCitizen'] == 1
            insights['senior_churn_rate'] = (
                (self.df[senior_mask]['Churn'] == 'Yes').sum() / senior_mask.sum() * 100
            )
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        print(f"Total Customers: {insights['total_customers']:,}")
        print(f"Overall Churn Rate: {insights['churn_rate']:.2f}%")
        print(f"Average Tenure: {insights['avg_tenure']:.1f} months")
        print(f"Average Monthly Charges: ${insights['avg_monthly_charges']:.2f}")
        print("\nChurn by Contract Type:")
        for contract, rate in insights['contract_churn'].items():
            print(f"  {contract}: {rate:.2f}%")
        if insights['senior_churn_rate']:
            print(f"\nSenior Citizen Churn Rate: {insights['senior_churn_rate']:.2f}%")
        
        return insights
    
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        self.df_processed = self.df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling Missing Values...")
        if 'TotalCharges' in self.df_processed.columns:
            self.df_processed['TotalCharges'] = pd.to_numeric(
                self.df_processed['TotalCharges'], errors='coerce'
            )
            self.df_processed['TotalCharges'].fillna(
                self.df_processed['MonthlyCharges'], inplace=True
            )
        
        # 2. Remove ID column
        if 'customerID' in self.df_processed.columns:
            self.df_processed.drop('customerID', axis=1, inplace=True)
        
        # 3. Feature Engineering
        print("2. Feature Engineering...")
        
        # Tenure groups
        self.df_processed['tenure_group'] = pd.cut(
            self.df_processed['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-12', '12-24', '24-48', '48-72']
        )
        
        # Charges per tenure
        self.df_processed['charges_per_tenure'] = (
            self.df_processed['MonthlyCharges'] / (self.df_processed['tenure'] + 1)
        )
        
        # Service count
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        self.df_processed['total_services'] = 0
        for col in service_cols:
            if col in self.df_processed.columns:
                self.df_processed['total_services'] += (
                    (self.df_processed[col] == 'Yes') | 
                    (self.df_processed[col] == 'Yes')
                ).astype(int)
        
        # 4. Encode categorical variables
        print("3. Encoding Categorical Variables...")
        
        # Separate target
        y = self.df_processed['Churn'].map({'No': 0, 'Yes': 1})
        X = self.df_processed.drop('Churn', axis=1)
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # 5. Train-test split
        print("4. Splitting Data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        
        # 6. Feature scaling
        print("5. Scaling Features...")
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        self.feature_names = list(X.columns)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Class distribution (train): {dict(self.y_train.value_counts())}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models for comparison"""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=15, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=5
            )
        }
        
        results = []
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            results.append(metrics)
            self.models[name] = model
            
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  F1-Score: {metrics['F1-Score']:.4f}")
            print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
        # Results comparison
        self.results_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(self.results_df.to_string(index=False))
        
        # Select best model based on F1-score
        best_idx = self.results_df['F1-Score'].idxmax()
        best_model_name = self.results_df.loc[best_idx, 'Model']
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        
        return self.results_df
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Hyperparameter tuning using Grid Search"""
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING - {model_name}")
        print("="*80)
        
        from sklearn.model_selection import GridSearchCV
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            base_model = GradientBoostingClassifier(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not configured for {model_name}")
            return None
        
        print(f"\nSearching best parameters...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"{model_name} (Tuned)"
        self.models[self.best_model_name] = self.best_model
        
        return grid_search.best_params_
    
    def evaluate_model(self, save_plots=True):
        """Comprehensive model evaluation"""
        print("\n" + "="*80)
        print(f"MODEL EVALUATION - {self.best_model_name}")
        print("="*80)
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # 1. Classification Report
        print("\n1. Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))
        
        # 2. Confusion Matrix
        print("\n2. Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # 3. Key Metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
            'Average Precision': average_precision_score(self.y_test, y_pred_proba)
        }
        
        print("\n3. Performance Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        if save_plots:
            self._create_evaluation_plots(y_pred, y_pred_proba, cm)
        
        # Cross-validation
        print("\n4. Cross-Validation Scores:")
        cv_scores = cross_val_score(
            self.best_model, self.X_train, self.y_train, 
            cv=5, scoring='f1'
        )
        print(f"   CV F1-Scores: {cv_scores}")
        print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return metrics
    
    def _create_evaluation_plots(self, y_pred, y_pred_proba, cm):
        """Create evaluation visualizations"""
        print("\nGenerating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                       label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        axes[1, 0].plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc="lower left")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            axes[1, 1].barh(range(len(indices)), importances[indices], color='skyblue')
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("   Saved: model_evaluation.png")
        plt.close()
    
    def explain_model(self):
        """Model explainability using permutation importance and partial dependence"""
        print("\n" + "="*80)
        print("MODEL EXPLAINABILITY")
        print("="*80)
        
        from sklearn.inspection import permutation_importance, PartialDependenceDisplay
        
        # 1. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\n1. Feature Importance (Built-in):")
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n   Top 10 Features:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"   {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
            
            # Plot
            plt.figure(figsize=(12, 8))
            top_n = 15
            top_indices = indices[:top_n]
            plt.barh(range(top_n), importances[top_indices], color='steelblue')
            plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Feature Importance (Built-in)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('/home/claude/feature_importance_builtin.png', dpi=300, bbox_inches='tight')
            print("\n   Saved: feature_importance_builtin.png")
            plt.close()
        
        # 2. Permutation Importance
        print("\n2. Permutation Importance:")
        print("   Computing... (this may take a moment)")
        
        perm_importance = permutation_importance(
            self.best_model, self.X_test, self.y_test,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
        
        print("\n   Top 10 Features (Permutation):")
        for i in range(min(10, len(perm_indices))):
            idx = perm_indices[i]
            print(f"   {i+1}. {self.feature_names[idx]}: "
                  f"{perm_importance.importances_mean[idx]:.4f} "
                  f"(+/- {perm_importance.importances_std[idx]:.4f})")
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_n = 15
        top_indices = perm_indices[:top_n]
        plt.barh(range(top_n), perm_importance.importances_mean[top_indices], 
                color='coral', xerr=perm_importance.importances_std[top_indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
        plt.xlabel('Permutation Importance', fontsize=12)
        plt.title('Permutation Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/home/claude/feature_importance_permutation.png', dpi=300, bbox_inches='tight')
        print("\n   Saved: feature_importance_permutation.png")
        plt.close()
        
        # 3. Partial Dependence Plots
        print("\n3. Partial Dependence Plots:")
        print("   Generating for top features...")
        
        # Select top features
        if hasattr(self.best_model, 'feature_importances_'):
            top_features = [self.feature_names[i] for i in indices[:4]]
        else:
            top_features = [self.feature_names[i] for i in perm_indices[:4]]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        PartialDependenceDisplay.from_estimator(
            self.best_model, self.X_test, top_features,
            ax=ax, n_cols=2, grid_resolution=50
        )
        plt.suptitle('Partial Dependence Plots - Top Features', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('/home/claude/partial_dependence.png', dpi=300, bbox_inches='tight')
        print("   Saved: partial_dependence.png")
        plt.close()
        
        return {
            'feature_importance': importances if hasattr(self.best_model, 'feature_importances_') else None,
            'permutation_importance': perm_importance.importances_mean,
            'top_features': top_features
        }
    
    def predict_single(self, customer_data):
        """Make prediction for a single customer"""
        # Convert to DataFrame
        if isinstance(customer_data, dict):
            df_input = pd.DataFrame([customer_data])
        else:
            df_input = customer_data.copy()
        
        # Remove customerID if present
        if 'customerID' in df_input.columns:
            df_input = df_input.drop('customerID', axis=1)
        
        # Feature engineering
        if 'tenure_group' not in df_input.columns and 'tenure' in df_input.columns:
            df_input['tenure_group'] = pd.cut(
                df_input['tenure'], 
                bins=[0, 12, 24, 48, 72], 
                labels=['0-12', '12-24', '24-48', '48-72']
            )
        
        if 'charges_per_tenure' not in df_input.columns:
            df_input['charges_per_tenure'] = (
                df_input['MonthlyCharges'] / (df_input['tenure'] + 1)
            )
        
        # Service count
        if 'total_services' not in df_input.columns:
            service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                           'StreamingTV', 'StreamingMovies']
            df_input['total_services'] = 0
            for col in service_cols:
                if col in df_input.columns:
                    df_input['total_services'] += (
                        (df_input[col] == 'Yes') | 
                        (df_input[col] == 'Yes')
                    ).astype(int)
        
        # Encode categorical variables
        for col in df_input.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                df_input[col] = self.label_encoders[col].transform(df_input[col].astype(str))
            else:
                # Handle new categories
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Reorder columns
        df_input = df_input[self.feature_names]
        
        # Scale numerical features
        numerical_cols = df_input.select_dtypes(include=[np.number]).columns
        df_input[numerical_cols] = self.scaler.transform(df_input[numerical_cols])
        
        # Predict
        prediction = self.best_model.predict(df_input)[0]
        probability = self.best_model.predict_proba(df_input)[0]
        
        result = {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability_no_churn': probability[0],
            'probability_churn': probability[1],
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    def _get_risk_level(self, churn_prob):
        """Determine risk level based on churn probability"""
        if churn_prob < 0.3:
            return 'Low'
        elif churn_prob < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def save_model(self, filepath='/home/claude/churn_model.pkl'):
        """Save trained model and preprocessing objects"""
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n✓ Model saved to: {filepath}")
        
        return filepath
    
    def load_model(self, filepath='/home/claude/churn_model.pkl'):
        """Load trained model and preprocessing objects"""
        model_package = joblib.load(filepath)
        
        self.best_model = model_package['model']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.feature_names = model_package['feature_names']
        self.best_model_name = model_package['model_name']
        
        print(f"\n✓ Model loaded from: {filepath}")
        
        return self
    
    def generate_report(self, save_path='/home/claude/churn_analysis_report.txt'):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("GENERATING ANALYSIS REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("TELECOM CUSTOMER CHURN PREDICTION - ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: {len(self.df)} customers")
        report.append(f"Features: {len(self.feature_names)}")
        report.append(f"Best Model: {self.best_model_name}")
        
        # Model performance
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        report.append("\n" + "-"*80)
        report.append("MODEL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Accuracy:  {accuracy_score(self.y_test, y_pred):.4f}")
        report.append(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        report.append(f"Recall:    {recall_score(self.y_test, y_pred):.4f}")
        report.append(f"F1-Score:  {f1_score(self.y_test, y_pred):.4f}")
        report.append(f"ROC-AUC:   {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            report.append("\n" + "-"*80)
            report.append("TOP 10 IMPORTANT FEATURES")
            report.append("-"*80)
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(indices))):
                idx = indices[i]
                report.append(f"{i+1:2d}. {self.feature_names[idx]:30s} {importances[idx]:.4f}")
        
        # Business insights
        report.append("\n" + "-"*80)
        report.append("BUSINESS INSIGHTS")
        report.append("-"*80)
        report.append("\nKey Findings:")
        report.append("1. Month-to-month contracts show higher churn rates")
        report.append("2. Customers with shorter tenure are at higher risk")
        report.append("3. Higher monthly charges correlate with increased churn")
        report.append("4. Lack of tech support increases churn probability")
        
        report.append("\nRecommendations:")
        report.append("1. Offer incentives for long-term contracts")
        report.append("2. Enhanced onboarding for new customers")
        report.append("3. Proactive outreach to high-risk customers")
        report.append("4. Improve tech support availability")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {save_path}")
        print("\n" + report_text)
        
        return report_text


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TELECOM CUSTOMER CHURN PREDICTION - COMPLETE ML PIPELINE")
    print("="*80)
    
    # Initialize predictor
    predictor = TelecomChurnPredictor()
    
    # Load data
    predictor.load_data()
    
    # EDA
    insights = predictor.exploratory_data_analysis()
    
    # Preprocessing
    predictor.preprocess_data()
    
    # Train models
    results = predictor.train_models()
    
    # Hyperparameter tuning
    best_params = predictor.hyperparameter_tuning('Random Forest')
    
    # Evaluation
    metrics = predictor.evaluate_model()
    
    # Explainability
    explainability = predictor.explain_model()
    
    # Save model
    model_path = predictor.save_model()
    
    # Generate report
    predictor.generate_report()
    
    # Example prediction
    print("\n" + "="*80)
    print("SAMPLE PREDICTION")
    print("="*80)
    
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Mailed check',
        'MonthlyCharges': 53.85,
        'TotalCharges': 108.15
    }
    
    result = predictor.predict_single(sample_customer)
    
    print("\nCustomer Profile:")
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction Results:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Churn Probability: {result['probability_churn']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - eda_overview.png")
    print("  - correlation_heatmap.png")
    print("  - categorical_analysis.png")
    print("  - model_evaluation.png")
    print("  - feature_importance_builtin.png")
    print("  - feature_importance_permutation.png")
    print("  - partial_dependence.png")
    print("  - churn_model.pkl")
    print("  - churn_analysis_report.txt")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
