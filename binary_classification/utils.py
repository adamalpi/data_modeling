import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_fscore_support

def load_preprocessed_data(file_path='data/preprocessed_data.parquet'):
    """Loads the preprocessed data from a Parquet file."""
    print(f"Loading preprocessed data from {file_path}...")
    try:
        df = pd.read_parquet(file_path)
        print("Data loaded successfully.")
        df.info()
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure '1_consolidate_data.ipynb' has been run.")
        raise
    except Exception as e:
        print(f"\nAn error occurred while loading the Parquet file: {e}")
        raise

def split_data_features_target(df):
    """Separates train/test sets and features/target from the dataframe."""
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    X_train = train_df.drop(['Class', 'split'], axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop(['Class', 'split'], axis=1)
    y_test = test_df['Class']
    
    print(f"Training features shape: {X_train.shape}, Training target shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}, Test target shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def convert_target_variable(y_series):
    """Converts a target variable series from object ('n'/'y') to numeric (0/1)."""
    if y_series.dtype == 'object':
        print("\nConverting target variable 'Class' to numeric (n=0, y=1)...")
        y_converted = y_series.map({'n': 0, 'y': 1})
        print("Target variable converted.")
        print("Value counts:\n", y_converted.value_counts())
        return y_converted
    elif y_series.dtype in ['int64', 'int32', 'float64', 'float32']:
        print("\nTarget variable 'Class' is already numeric.")
        return y_series
    else:
        print(f"\nTarget variable 'Class' is of unexpected type: {y_series.dtype}. No conversion applied.")
        return y_series

def evaluate_model_performance(y_true, y_pred, X_features_for_proba, model, model_name):
    """
    Calculates, prints, and plots evaluation metrics for a binary classifier.
    Includes Accuracy, Classification Report, Confusion Matrix, and ROC AUC.
    """
    print(f"\n--- {model_name} Evaluation ---")

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    target_names = ['Class n (0)', 'Class y (1)'] if np.all(np.isin(np.unique(y_true), [0, 1])) else None
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    classes = model.classes_ if hasattr(model, 'classes_') else [0, 1] # Handle cases where model might not have .classes_
    # Ensure labels match the actual unique values in y_true and y_pred if possible
    # For simplicity, using model.classes_ or default [0,1]
    # More robust: unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # ROC Curve and AUC
    if hasattr(model, "predict_proba") and X_features_for_proba is not None:
        # Ensure X_features_for_proba is a DataFrame for consistent column access if model expects it
        if not isinstance(X_features_for_proba, pd.DataFrame) and hasattr(X_features_for_proba, 'columns'):
             X_features_for_proba = pd.DataFrame(X_features_for_proba, columns=model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None)


        y_pred_proba = model.predict_proba(X_features_for_proba)[:, 1]
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("\nROC Curve not available: Model lacks predict_proba or features for probabilities not provided.")

def plot_threshold_tuning_curves(threshold_values, metrics_dict, class_label_name, model_name_suffix=""):
    """
    Plots precision, recall, and F1-score against threshold values for a specific class.
    metrics_dict should be a dictionary like:
    {'precision': [...], 'recall': [...], 'f1': [...]}
    """
    plt.figure(figsize=(12, 7))
    plt.plot(threshold_values, metrics_dict['precision'], label=f'Precision ({class_label_name})', marker='o')
    plt.plot(threshold_values, metrics_dict['recall'], label=f'Recall ({class_label_name})', marker='x')
    plt.plot(threshold_values, metrics_dict['f1'], label=f'F1-score ({class_label_name})', marker='s')
    plt.title(f'Precision, Recall, and F1-score for {class_label_name} vs. Threshold {model_name_suffix}')
    plt.xlabel('Threshold (Probability for Positive Class)')
    plt.ylabel('Score')
    plt.xticks(np.round(threshold_values,2))
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_feature_importance(model, feature_names, top_n=20):
    """Plots feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have 'feature_importances_' attribute. Cannot plot feature importance.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print(f"\n--- Feature Importance Analysis (Top {min(top_n, len(feature_importance_df))}) ---")
    print(feature_importance_df.head(top_n))
    
    plt.figure(figsize=(12, 8))
    num_features_to_plot = min(len(feature_importance_df), top_n)
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(num_features_to_plot), palette='viridis')
    plt.title(f'Top {num_features_to_plot} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def perform_shap_analysis(model, X_data, model_type='tree'):
    """Performs and plots SHAP analysis (summary and force plot for the first instance)."""
    try:
        import shap
        shap.initjs() # Initialize JavaScript visualization in the notebook
    except ImportError:
        print("SHAP library not found. Please install it to run SHAP analysis (e.g., pip install shap).")
        return

    print("\n--- SHAP Value Analysis ---")
    
    if model_type == 'tree' and hasattr(shap, 'TreeExplainer'):
        explainer = shap.TreeExplainer(model)
    elif hasattr(shap, 'KernelExplainer'): # Fallback or for non-tree models
        # KernelExplainer can be slow, especially for large X_data.
        # Consider using a subset of X_data for background data if performance is an issue.
        # X_data_summary = shap.sample(X_data, 100) # Example: use 100 samples for summary
        explainer = shap.KernelExplainer(model.predict_proba, X_data) # or model.predict for regression
    else:
        print("Could not determine appropriate SHAP explainer for the model.")
        return

    print("Calculating SHAP values...")
    # For KernelExplainer, shap_values might need X_data itself if not explaining predictions on X_data
    shap_values = explainer.shap_values(X_data) 
    print("SHAP values calculated.")

    # SHAP Summary Plot (Beeswarm)
    print("\nSHAP Summary Plot (Beeswarm):")
    # For binary classification, shap_values from TreeExplainer might be a list of two arrays.
    # We usually plot for the positive class.
    shap_values_for_plot = shap_values
    if isinstance(shap_values, list) and len(shap_values) == 2: # Common for binary classification
        shap_values_for_plot = shap_values[1] # SHAP values for the positive class (class 1)
    
    # Ensure X_data is a DataFrame for summary_plot if it expects column names
    if not isinstance(X_data, pd.DataFrame) and hasattr(X_data, 'columns'):
        X_data_df = pd.DataFrame(X_data, columns=model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None)
    else:
        X_data_df = X_data

    shap.summary_plot(shap_values_for_plot, X_data_df, plot_type="beeswarm")

    # SHAP Force Plot for the first instance
    if len(X_data_df) > 0:
        print("\nSHAP Force Plot for the first test instance:")
        expected_value_to_use = explainer.expected_value
        shap_values_instance = shap_values_for_plot

        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
            expected_value_to_use = explainer.expected_value[1] # for the positive class
        
        # If shap_values_for_plot is 2D (instances, features)
        if len(shap_values_instance.shape) == 2:
            shap_values_instance = shap_values_instance[0,:] # first instance

        shap.force_plot(expected_value_to_use, 
                        shap_values_instance, 
                        X_data_df.iloc[0,:], 
                        matplotlib=True)
    else:
        print("Data is empty, cannot generate force plot for an instance.")