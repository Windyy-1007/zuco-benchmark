"""
Convolutional Neural Network (CNN) Classifier for Dyslexia Prediction

This module implements a CNN for dyslexia prediction using EEG and eye-tracking features.
The CNN is designed to capture spatial and temporal patterns in neural data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config

# Set random seeds for reproducibility
tf.random.set_seed(config.seed)
np.random.seed(config.seed)

class CNNDyslexiaClassifier:
    """
    CNN classifier for dyslexia prediction
    """
    
    def __init__(self, cnn_config=None):
        self.cnn_config = cnn_config or config.cnn_config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def _reshape_features_for_cnn(self, X):
        """
        Reshape features for CNN input
        For EEG data: reshape to (samples, electrodes, time_windows, 1)
        For combined data: reshape to (samples, features, 1, 1) or (samples, sqrt(features), sqrt(features), 1)
        """
        if len(X.shape) == 2:
            # Determine the best reshape strategy based on feature count
            n_features = X.shape[1]
            
            if n_features == 420:  # electrode_features_all (105 electrodes * 4 frequency bands)
                # Reshape to (samples, electrodes, frequency_bands, 1)
                X_reshaped = X.reshape(-1, 105, 4, 1)
            elif n_features >= 100:
                # For large feature sets, create a 2D spatial representation
                sqrt_features = int(np.sqrt(n_features))
                if sqrt_features * sqrt_features == n_features:
                    X_reshaped = X.reshape(-1, sqrt_features, sqrt_features, 1)
                else:
                    # Pad to make it square
                    next_square = int(np.ceil(np.sqrt(n_features))) ** 2
                    X_padded = np.pad(X, ((0, 0), (0, next_square - n_features)), mode='constant')
                    sqrt_size = int(np.sqrt(next_square))
                    X_reshaped = X_padded.reshape(-1, sqrt_size, sqrt_size, 1)
            else:
                # For small feature sets, use 1D convolution
                X_reshaped = X.reshape(-1, n_features, 1, 1)
                
        else:
            X_reshaped = X
            
        return X_reshaped
    
    def _build_model(self, input_shape):
        """
        Build CNN model architecture
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Convolutional layers
        for i, conv_config in enumerate(self.cnn_config["conv_layers"]):
            if len(input_shape) == 3:  # 2D convolution
                model.add(layers.Conv2D(
                    filters=conv_config["filters"],
                    kernel_size=conv_config["kernel_size"],
                    activation=conv_config["activation"],
                    padding='same',
                    name=f'conv2d_{i+1}'
                ))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
            else:  # 1D convolution
                model.add(layers.Conv1D(
                    filters=conv_config["filters"],
                    kernel_size=conv_config["kernel_size"],
                    activation=conv_config["activation"],
                    padding='same',
                    name=f'conv1d_{i+1}'
                ))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D(pool_size=2, padding='same'))
            
            model.add(layers.Dropout(self.cnn_config["dropout_rate"]))
        
        # Global pooling to reduce dimensions
        if len(input_shape) == 3:
            model.add(layers.GlobalAveragePooling2D())
        else:
            model.add(layers.GlobalAveragePooling1D())
        
        # Dense layers
        for dense_units in self.cnn_config["dense_layers"]:
            model.add(layers.Dense(dense_units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.cnn_config["dropout_rate"]))
        
        # Output layer
        model.add(layers.Dense(config.n_classes, activation='softmax' if config.n_classes > 2 else 'sigmoid'))
        
        return model
    
    def train(self, X, y, validation_data=None):
        """
        Train the CNN model
        """
        print("\nTraining CNN model...")
        
        # Reshape features for CNN
        X_reshaped = self._reshape_features_for_cnn(X)
        print(f"Input shape after reshaping: {X_reshaped.shape}")
        
        # Scale features
        original_shape = X_reshaped.shape
        X_scaled = self.scaler.fit_transform(X_reshaped.reshape(X_reshaped.shape[0], -1))
        X_scaled = X_scaled.reshape(original_shape)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        if config.n_classes > 2:
            y_encoded = keras.utils.to_categorical(y_encoded, num_classes=config.n_classes)
        
        # Split validation data if not provided
        if validation_data is None and self.cnn_config["validation_split"] > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded,
                test_size=self.cnn_config["validation_split"],
                random_state=config.seed,
                stratify=y
            )
        else:
            X_train, y_train = X_scaled, y_encoded
            X_val, y_val = validation_data if validation_data else (None, None)
        
        # Build model
        input_shape = X_train.shape[1:]
        self.model = self._build_model(input_shape)
        
        # Compile model
        if config.n_classes > 2:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cnn_config["learning_rate"]),
            loss=loss,
            metrics=metrics
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Set up callbacks
        callbacks = []
        if self.cnn_config["early_stopping_patience"] > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.cnn_config["early_stopping_patience"],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train model
        validation_data_tuple = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.cnn_config["epochs"],
            batch_size=self.cnn_config["batch_size"],
            validation_data=validation_data_tuple,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate training performance
        train_pred = self.model.predict(X_train)
        if config.n_classes > 2:
            train_pred_classes = np.argmax(train_pred, axis=1)
            train_true_classes = np.argmax(y_train, axis=1)
        else:
            train_pred_classes = (train_pred > 0.5).astype(int).flatten()
            train_true_classes = y_train
            
        train_accuracy = accuracy_score(train_true_classes, train_pred_classes)
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained CNN
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Reshape and scale features
        X_reshaped = self._reshape_features_for_cnn(X)
        original_shape = X_reshaped.shape
        X_scaled = self.scaler.transform(X_reshaped.reshape(X_reshaped.shape[0], -1))
        X_scaled = X_scaled.reshape(original_shape)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        if config.n_classes > 2:
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        return predicted_classes
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()

def build_data_cnn(data, labels):
    """
    Build data for CNN classification
    """
    X, y = [], []
    for subj in data:
        for features in data[subj]:
            X.append(features)
        for label in labels[subj]:
            if config.task_type == "dyslexia_prediction":
                y.append(int(label))
            else:
                y.append(1 if label == "NR" else 0)
    return np.array(X), np.array(y)

def predict_subject_cnn(cnn_classifier, test_X, test_y, index, subj):
    """
    Predict on a single subject using CNN
    """
    print(f"\nPredicting on subject {subj}")
    prediction = cnn_classifier.predict(test_X[index])
    
    if len(test_y[index]) > 0:  # If we have true labels
        accuracy = accuracy_score(test_y[index], prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y[index], prediction, average='binary', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        
        return [prediction], accuracy, f1, precision, recall
    else:
        return [prediction], 0, 0, 0, 0

def benchmark_cnn(X, y, test_X, test_y):
    """
    CNN benchmark for dyslexia prediction
    """
    print("\n" + "="*50)
    print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("="*50)
    
    # Initialize CNN classifier
    cnn_classifier = CNNDyslexiaClassifier()
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_cnn(X, y)
    
    # Build test data
    builded = zip(*[build_data_cnn({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Training features: {train_X.shape[1]}")
    print(f"Training class distribution: {np.bincount(train_y)}")
    
    # Train CNN
    history = cnn_classifier.train(train_X, train_y)
    
    # Plot training history
    cnn_classifier.plot_training_history(save_path="../results/cnn_training_history.png")
    
    # Predict on all test subjects
    results = zip(*[predict_subject_cnn(cnn_classifier, test_X_array, test_y_array, 
                                       index, subj)
                   for index, subj in enumerate(config.heldout_subjects)])
    results = list(map(list, results))
    
    # No bootstrap for CNN (takes too long)
    bootstrap_results = [[]]
    
    return [results] + [bootstrap_results]

def benchmark_cnn_baseline(X, y, test_X, test_y):
    """
    CNN baseline (no evaluation)
    """
    print("\n" + "="*50)
    print("CNN BASELINE")
    print("="*50)
    
    # Initialize CNN classifier
    cnn_classifier = CNNDyslexiaClassifier()
    
    print("\nPre-processing data...")
    train_X, train_y = build_data_cnn(X, y)
    
    # Build test data
    builded = zip(*[build_data_cnn({subj: test_X[subj]}, {subj: test_y[subj]})
                   for subj in test_X])
    builded = list(map(list, builded))
    test_X_array, test_y_array = builded
    
    train_X, train_y = shuffle(train_X, train_y)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Training features: {train_X.shape[1]}")
    
    # Train CNN
    cnn_classifier.train(train_X, train_y)
    
    # Plot training history
    cnn_classifier.plot_training_history(save_path="../results/cnn_training_history.png")
    
    # Generate predictions
    results = []
    for index, subj in enumerate(config.heldout_subjects):
        print(f"\nPredicting on subject {subj}")
        prediction = cnn_classifier.predict(test_X_array[index])
        results.append(prediction)
        print(f"Generated {len(prediction)} predictions")
    
    return results

if __name__ == "__main__":
    print("CNN Classifier for Dyslexia Prediction")
    print("Use this module through the main benchmark scripts.")
