#!/usr/bin/env python3
"""
Fall Detection Model Evaluation Script

This script loads all trained models and evaluates them on validation and test datasets.
Focuses on fall detection performance with comprehensive metrics:
- Accuracy
- Precision 
- Recall
- F1-Score

Generates evaluation visualizations saved to visualizations/model_evaluation/ folder.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any

# Machine Learning
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def setup_directories() -> Path:
    """Create and return the model evaluation directory."""
    eval_dir = Path('visualizations/model_evaluation')
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def load_test_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load preprocessed test data from pickle files.
    
    Returns:
        Tuple of (traditional_data, sequence_data) dictionaries
    """
    print("Loading test data...")
    
    # Load the split datasets
    with open('data/processed/fall_detection_X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/fall_detection_X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('data/processed/fall_detection_X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/processed/fall_detection_y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/processed/fall_detection_y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('data/processed/fall_detection_y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Load preprocessing artifacts
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Data loaded successfully!")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Encode labels
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare traditional ML data
    traditional_data = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_encoded,
        'y_val': y_val_encoded,
        'y_test': y_test_encoded,
        'y_train_labels': y_train,
        'y_val_labels': y_val,
        'y_test_labels': y_test,
        'label_encoder': le
    }
    
    # Prepare sequence data for deep learning
    n_timesteps = 1
    X_train_seq = X_train.values.reshape(X_train.shape[0], n_timesteps, X_train.shape[1])
    X_val_seq = X_val.values.reshape(X_val.shape[0], n_timesteps, X_val.shape[1])
    X_test_seq = X_test.values.reshape(X_test.shape[0], n_timesteps, X_test.shape[1])
    
    sequence_data = {
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': y_train_encoded,
        'y_val': y_val_encoded,
        'y_test': y_test_encoded,
        'y_train_labels': y_train,
        'y_val_labels': y_val,
        'y_test_labels': y_test,
        'label_encoder': le
    }
    
    return traditional_data, sequence_data


def load_traditional_model(model_path: Path) -> Any:
    """Load a traditional ML model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"⚠ Error loading {model_path}: {e}")
        return None


def load_deep_learning_model(model_path: Path) -> Any:
    """Load a deep learning model from h5 file."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"⚠ Error loading {model_path}: {e}")
        return None


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
    """Calculate comprehensive metrics for model evaluation."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def evaluate_traditional_model(model: Any, model_name: str, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Evaluate a traditional ML model on validation and test sets."""
    if model is None:
        return {}
    
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    val_pred = model.predict(data['X_val'])
    test_pred = model.predict(data['X_test'])
    
    # Calculate metrics
    val_metrics = calculate_metrics(data['y_val'], val_pred)
    test_metrics = calculate_metrics(data['y_test'], test_pred)
    
    print(f"  Validation - Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1_score']:.3f}")
    print(f"  Test - Acc: {test_metrics['accuracy']:.3f}, F1: {test_metrics['f1_score']:.3f}")
    
    return {
        'validation': val_metrics,
        'test': test_metrics,
        'val_predictions': val_pred,
        'test_predictions': test_pred
    }


def evaluate_deep_learning_model(model: Any, model_name: str, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Evaluate a deep learning model on validation and test sets."""
    if model is None:
        return {}
    
    print(f"\nEvaluating {model_name}...")
    
    # Predictions (probabilities)
    val_pred_proba = model.predict(data['X_val'], verbose=0)
    test_pred_proba = model.predict(data['X_test'], verbose=0)
    
    # Convert to class predictions
    val_pred = np.argmax(val_pred_proba, axis=1)
    test_pred = np.argmax(test_pred_proba, axis=1)
    
    # Calculate metrics
    val_metrics = calculate_metrics(data['y_val'], val_pred)
    test_metrics = calculate_metrics(data['y_test'], test_pred)
    
    print(f"  Validation - Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1_score']:.3f}")
    print(f"  Test - Acc: {test_metrics['accuracy']:.3f}, F1: {test_metrics['f1_score']:.3f}")
    
    return {
        'validation': val_metrics,
        'test': test_metrics,
        'val_predictions': val_pred,
        'test_predictions': test_pred
    }


def create_metrics_comparison_plot(results: Dict[str, Dict], eval_dir: Path) -> None:
    """Create comprehensive metrics comparison plot."""
    # Prepare data for plotting
    models = []
    datasets = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, model_results in results.items():
        if not model_results:  # Skip empty results
            continue
            
        for dataset_name, metrics in model_results.items():
            if dataset_name in ['validation', 'test']:
                models.append(model_name)
                datasets.append(dataset_name.title())
                accuracies.append(metrics['accuracy'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Dataset': datasets,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison - Fall Detection', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Pivot data for better visualization
        pivot_df = df.pivot(index='Model', columns='Dataset', values=metric)
        
        # Create grouped bar plot
        pivot_df.plot(kind='bar', ax=ax, alpha=0.8, width=0.8)
        ax.set_title(f'{metric} by Model and Dataset', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = eval_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics comparison saved to {output_path}")


def create_confusion_matrices(results: Dict[str, Dict], data: Dict[str, Any], eval_dir: Path) -> None:
    """Create confusion matrices for all models on test set."""
    class_names = data['label_encoder'].classes_
    
    # Filter models with results
    valid_models = {k: v for k, v in results.items() if v and 'test_predictions' in v}
    
    n_models = len(valid_models)
    if n_models == 0:
        return
    
    # Calculate grid size
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('Confusion Matrices - Test Set Performance', fontsize=16, fontweight='bold')
    
    # Ensure axes is always a list
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for idx, (model_name, model_results) in enumerate(valid_models.items()):
        ax = axes[idx]
        
        # Get predictions
        y_true = data['y_test']
        y_pred = model_results['test_predictions']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = eval_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrices saved to {output_path}")


def create_fall_detection_focus_plot(results: Dict[str, Dict], data: Dict[str, Any], eval_dir: Path) -> None:
    """Create focused analysis on fall detection performance."""
    class_names = data['label_encoder'].classes_
    
    # Find the index of 'Falls' class
    try:
        fall_class_idx = np.where(class_names == 'Falls')[0][0]
    except IndexError:
        print("⚠ 'Falls' class not found in label encoder classes")
        return
    
    # Prepare data for fall detection metrics
    models = []
    datasets = []
    fall_precision = []
    fall_recall = []
    fall_f1 = []
    
    for model_name, model_results in results.items():
        if not model_results or 'test_predictions' not in model_results:
            continue
            
        for dataset_name in ['validation', 'test']:
            if dataset_name not in model_results:
                continue
                
            # Get true labels and predictions
            if dataset_name == 'validation':
                y_true = data['y_val']
                y_pred = model_results['val_predictions']
            else:
                y_true = data['y_test']
                y_pred = model_results['test_predictions']
            
            # Calculate class-specific metrics for Falls
            precision = precision_score(y_true, y_pred, labels=[fall_class_idx], average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, labels=[fall_class_idx], average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=[fall_class_idx], average=None, zero_division=0)
            
            models.append(model_name)
            datasets.append(dataset_name.title())
            fall_precision.append(precision[0] if len(precision) > 0 else 0)
            fall_recall.append(recall[0] if len(recall) > 0 else 0)
            fall_f1.append(f1[0] if len(f1) > 0 else 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Dataset': datasets,
        'Fall_Precision': fall_precision,
        'Fall_Recall': fall_recall,
        'Fall_F1': fall_f1
    })
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Fall Detection Specific Performance', fontsize=16, fontweight='bold')
    
    metrics = ['Fall_Precision', 'Fall_Recall', 'Fall_F1']
    titles = ['Precision (Falls)', 'Recall (Falls)', 'F1-Score (Falls)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Pivot data
        pivot_df = df.pivot(index='Model', columns='Dataset', values=metric)
        
        # Create bar plot
        pivot_df.plot(kind='bar', ax=ax, alpha=0.8, width=0.8, color=['skyblue', 'lightcoral'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = eval_dir / 'fall_detection_focus.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Fall detection focus analysis saved to {output_path}")


def create_model_ranking_plot(results: Dict[str, Dict], eval_dir: Path) -> None:
    """Create model ranking visualization based on test performance."""
    # Prepare data
    model_scores = []
    
    for model_name, model_results in results.items():
        if not model_results or 'test' not in model_results:
            continue
            
        test_metrics = model_results['test']
        model_scores.append({
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1-Score': test_metrics['f1_score'],
            'Average': np.mean([test_metrics['accuracy'], test_metrics['precision'], 
                              test_metrics['recall'], test_metrics['f1_score']])
        })
    
    if not model_scores:
        return
    
    # Create DataFrame and sort by average score
    df = pd.DataFrame(model_scores)
    df = df.sort_values('Average', ascending=True)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.barh(x + i * width, df[metric], width, label=metric, color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width_val:.3f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Score')
    ax.set_title('Model Performance Ranking - Test Set', fontweight='bold', fontsize=14)
    ax.set_yticks(x + width * 1.5)
    ax.set_yticklabels(df['Model'])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1.1])
    
    plt.tight_layout()
    
    # Save plot
    output_path = eval_dir / 'model_ranking.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model ranking saved to {output_path}")


def print_detailed_results(results: Dict[str, Dict]) -> None:
    """Print detailed evaluation results."""
    print(f"\n" + "="*80)
    print("DETAILED MODEL EVALUATION RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        if not model_results:
            continue
            
        print(f"\n{model_name.upper()}")
        print("-" * 50)
        
        for dataset_name in ['validation', 'test']:
            if dataset_name in model_results:
                metrics = model_results[dataset_name]
                print(f"{dataset_name.title()} Set:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                print()


def main():
    """Main evaluation function."""
    print("="*60)
    print("FALL DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Setup
    eval_dir = setup_directories()
    
    # Load data
    traditional_data, sequence_data = load_test_data()
    
    # Initialize results storage
    results = {}
    
    # Evaluate Traditional ML Models
    print(f"\n" + "="*50)
    print("EVALUATING TRADITIONAL ML MODELS")
    print("="*50)
    
    # XGBoost
    xgb_model = load_traditional_model(Path('models/xgboost_model.pkl'))
    results['XGBoost'] = evaluate_traditional_model(xgb_model, 'XGBoost', traditional_data)
    
    # Random Forest  
    rf_model = load_traditional_model(Path('models/random_forest_model.pkl'))
    results['Random Forest'] = evaluate_traditional_model(rf_model, 'Random Forest', traditional_data)
    
    # Evaluate Deep Learning Models
    print(f"\n" + "="*50)
    print("EVALUATING DEEP LEARNING MODELS")
    print("="*50)
    
    # CNN1D
    cnn1d_model = load_deep_learning_model(Path('models/cnn1d_best_model.h5'))
    results['CNN1D'] = evaluate_deep_learning_model(cnn1d_model, 'CNN1D', sequence_data)
    
    # RNN
    rnn_model = load_deep_learning_model(Path('models/rnn_best_model.h5'))
    results['RNN'] = evaluate_deep_learning_model(rnn_model, 'RNN', sequence_data)
    
    # LSTM
    lstm_model = load_deep_learning_model(Path('models/lstm_best_model.h5'))
    results['LSTM'] = evaluate_deep_learning_model(lstm_model, 'LSTM', sequence_data)
    
    # Generate Visualizations
    print(f"\n" + "="*50)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("="*50)
    
    create_metrics_comparison_plot(results, eval_dir)
    create_confusion_matrices(results, traditional_data, eval_dir)  # Use traditional_data for consistent labeling
    create_fall_detection_focus_plot(results, traditional_data, eval_dir)
    create_model_ranking_plot(results, eval_dir)
    
    # Print detailed results
    print_detailed_results(results)
    
    # Summary
    print(f"\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Find best model by test F1-score
    valid_results = {k: v for k, v in results.items() if v and 'test' in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['test']['f1_score'])
        print(f"🏆 Best Model (Test F1-Score): {best_model[0]} ({best_model[1]['test']['f1_score']:.4f})")
        
        best_accuracy = max(valid_results.items(), key=lambda x: x[1]['test']['accuracy'])
        print(f"🎯 Best Accuracy (Test): {best_accuracy[0]} ({best_accuracy[1]['test']['accuracy']:.4f})")
    
    print(f"\n✓ Evaluation completed successfully!")
    print(f"✓ All visualizations saved to: {eval_dir.absolute()}")


if __name__ == "__main__":
    main()
