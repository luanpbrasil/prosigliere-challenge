#!/usr/bin/env python3
"""
Fall Detection Model Training Script

This script trains and saves multiple machine learning models for fall detection:
- XGBoost: Gradient boosting classifier
- Random Forest: Ensemble tree-based classifier  
- CNN1D: 1D Convolutional Neural Network
- RNN: Recurrent Neural Network (Advanced Multi-layer)
- LSTM: Long Short-Term Memory Network

All models are trained on the same preprocessed dataset and saved to the models/ folder.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

# Traditional ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    SimpleRNN, BatchNormalization, Input, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Training configuration
CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'patience': 10,
    'learning_rate': 0.001,
    'random_state': 42
}


def setup_directories() -> Path:
    """Create and return the models directory."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    return models_dir


def load_training_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load preprocessed training data from pickle files.
    
    Returns:
        Tuple of (traditional_data, sequence_data) dictionaries
    """
    print("Loading preprocessed training data...")
    
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
    
    print(f"✓ Data loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    
    # Encode labels for traditional ML models
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    # Scale features for traditional ML
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare data for traditional ML models
    traditional_data = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled, 
        'X_test': X_test_scaled,
        'y_train': y_train_encoded,
        'y_val': y_val_encoded,
        'y_test': y_test_encoded,
        'label_encoder': le,
        'scaler': scaler
    }
    
    # Prepare sequence data for deep learning models
    n_timesteps = 1
    X_train_seq = X_train.values.reshape(X_train.shape[0], n_timesteps, X_train.shape[1])
    X_val_seq = X_val.values.reshape(X_val.shape[0], n_timesteps, X_val.shape[1])
    X_test_seq = X_test.values.reshape(X_test.shape[0], n_timesteps, X_test.shape[1])
    
    # Convert to categorical for deep learning
    y_train_cat = to_categorical(y_train_encoded, num_classes=3)
    y_val_cat = to_categorical(y_val_encoded, num_classes=3)
    y_test_cat = to_categorical(y_test_encoded, num_classes=3)
    
    sequence_data = {
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': y_train_cat,
        'y_val': y_val_cat,
        'y_test': y_test_cat,
        'input_shape': (n_timesteps, X_train.shape[1]),
        'num_classes': 3
    }
    
    print(f"✓ Data preprocessing completed!")
    print(f"  Traditional ML data: {X_train_scaled.shape}")
    print(f"  Sequence data: {X_train_seq.shape}")
    print(f"  Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return traditional_data, sequence_data


def train_xgboost(traditional_data: Dict[str, Any], models_dir: Path) -> Dict[str, float]:
    """Train and save XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("⚠ XGBoost not available, skipping...")
        return {}
    
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    # Create and train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=CONFIG['random_state'],
        eval_metric='mlogloss'
    )
    
    model.fit(traditional_data['X_train'], traditional_data['y_train'])
    
    # Evaluate
    train_pred = model.predict(traditional_data['X_train'])
    val_pred = model.predict(traditional_data['X_val'])
    test_pred = model.predict(traditional_data['X_test'])
    
    train_acc = accuracy_score(traditional_data['y_train'], train_pred)
    val_acc = accuracy_score(traditional_data['y_val'], val_pred)
    test_acc = accuracy_score(traditional_data['y_test'], test_pred)
    
    # Save model
    model_path = models_dir / 'xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ XGBoost model saved to {model_path}")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Validation Accuracy: {val_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc
    }


def train_random_forest(traditional_data: Dict[str, Any], models_dir: Path) -> Dict[str, float]:
    """Train and save Random Forest model."""
    print("\n" + "="*50)
    print("Training Random Forest Model")
    print("="*50)
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=CONFIG['random_state'],
        class_weight='balanced'
    )
    
    model.fit(traditional_data['X_train'], traditional_data['y_train'])
    
    # Evaluate
    train_pred = model.predict(traditional_data['X_train'])
    val_pred = model.predict(traditional_data['X_val'])
    test_pred = model.predict(traditional_data['X_test'])
    
    train_acc = accuracy_score(traditional_data['y_train'], train_pred)
    val_acc = accuracy_score(traditional_data['y_val'], val_pred)
    test_acc = accuracy_score(traditional_data['y_test'], test_pred)
    
    # Save model
    model_path = models_dir / 'random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Random Forest model saved to {model_path}")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Validation Accuracy: {val_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc
    }


def create_cnn1d_model(input_shape: Tuple[int, int], num_classes: int) -> Model:
    """Create CNN1D model architecture."""
    model = Sequential([
        # Dense layers instead of Conv1D for single timestep data
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_rnn_model(input_shape: Tuple[int, int], num_classes: int) -> Model:
    """Create Advanced RNN model architecture."""
    inputs = Input(shape=input_shape)
    
    # Multi-layer RNN with batch normalization and dropout
    x = inputs
    rnn_units = 128
    num_layers = 3
    dropout_rate = 0.3
    
    for i in range(num_layers):
        x = SimpleRNN(rnn_units, return_sequences=True if i < num_layers - 1 else False)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Additional dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(input_shape: Tuple[int, int], num_classes: int) -> Model:
    """Create LSTM model architecture.""" 
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_deep_learning_model(
    model: Model, 
    model_name: str,
    sequence_data: Dict[str, Any], 
    models_dir: Path
) -> Dict[str, float]:
    """Train and save a deep learning model."""
    print(f"\n" + "="*50)
    print(f"Training {model_name} Model")
    print("="*50)
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        ModelCheckpoint(
            filepath=str(models_dir / f'{model_name.lower()}_best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print(f"Training {model_name} for up to {CONFIG['epochs']} epochs...")
    history = model.fit(
        sequence_data['X_train'], sequence_data['y_train'],
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=(sequence_data['X_val'], sequence_data['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(sequence_data['X_train'], sequence_data['y_train'], verbose=0)
    val_loss, val_acc = model.evaluate(sequence_data['X_val'], sequence_data['y_val'], verbose=0)
    test_loss, test_acc = model.evaluate(sequence_data['X_test'], sequence_data['y_test'], verbose=0)
    
    # Save final model
    final_model_path = models_dir / f'{model_name.lower()}_final_model.h5'
    model.save(final_model_path)
    
    print(f"✓ {model_name} model saved to {final_model_path}")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Validation Accuracy: {val_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print(f"  Best model saved to: {models_dir / f'{model_name.lower()}_best_model.h5'}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'history': history
    }


def save_training_artifacts(traditional_data: Dict[str, Any], models_dir: Path) -> None:
    """Save label encoder and scaler for future use."""
    # Save label encoder
    le_path = models_dir / 'label_encoder.pkl'
    with open(le_path, 'wb') as f:
        pickle.dump(traditional_data['label_encoder'], f)
    
    # Save scaler
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(traditional_data['scaler'], f)
    
    print(f"✓ Training artifacts saved:")
    print(f"  Label encoder: {le_path}")
    print(f"  Scaler: {scaler_path}")


def print_results_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print summary of all model results."""
    print(f"\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 45)
    
    for model_name, metrics in results.items():
        if metrics:  # Skip empty results (e.g., if XGBoost unavailable)
            print(f"{model_name:<15} {metrics['train_acc']:.3f}   {metrics['val_acc']:.3f}   {metrics['test_acc']:.3f}")
    
    # Find best model by validation accuracy
    valid_results = {k: v for k, v in results.items() if v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['val_acc'])
        print(f"\n🏆 Best Model: {best_model[0]} (Val Acc: {best_model[1]['val_acc']:.3f})")


def main():
    """Main training function."""
    print("="*60)
    print("FALL DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    models_dir = setup_directories()
    
    # Load data
    traditional_data, sequence_data = load_training_data()
    
    # Results storage
    results = {}
    
    # Train traditional ML models
    results['XGBoost'] = train_xgboost(traditional_data, models_dir)
    results['Random Forest'] = train_random_forest(traditional_data, models_dir)
    
    # Train deep learning models
    # CNN1D
    cnn1d_model = create_cnn1d_model(sequence_data['input_shape'], sequence_data['num_classes'])
    results['CNN1D'] = train_deep_learning_model(cnn1d_model, 'CNN1D', sequence_data, models_dir)
    
    # RNN
    rnn_model = create_rnn_model(sequence_data['input_shape'], sequence_data['num_classes'])
    results['RNN'] = train_deep_learning_model(rnn_model, 'RNN', sequence_data, models_dir)
    
    # LSTM
    lstm_model = create_lstm_model(sequence_data['input_shape'], sequence_data['num_classes'])
    results['LSTM'] = train_deep_learning_model(lstm_model, 'LSTM', sequence_data, models_dir)
    
    # Save training artifacts
    save_training_artifacts(traditional_data, models_dir)
    
    # Print summary
    print_results_summary(results)
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ All models and artifacts saved to: {models_dir.absolute()}")


if __name__ == "__main__":
    main()
