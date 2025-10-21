#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import keras
import pickle

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, RepeatVector, Reshape, Cropping1D, Lambda, Multiply

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class VINModelPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 17 
        self.vocab_size = None


    def load_and_preprocess_data(self, csv_file_path):
        """
        Load and preprocess the VIN and Model data from CSV
        """
        # Load data
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        
        # Basic data cleaning
        df = df.dropna()  # Remove any rows with missing values
        df['vin'] = df['vin'].str.upper()  # Normalize VIN to uppercase
        df['model'] = df['model'].str.strip().replace("-", " ")  # Remove whitespace from model names
        df['model'] = df['model'].str.lower()  # to lowercase
        # df['model'] = df['model'].apply(unify_brand_name)  # Unify model names
        
        # Validate VIN format (should be 17 characters)
        df = df[df['vin'].str.len() == 17]
        df = df[df['model'] != '']  # Remove empty model names

        print(f"After cleaning: {len(df)} records")
        
        # Display model distribution
        print("Model distribution:")
        valid_groups = df['model'].value_counts()
        print(valid_groups)
        min_group_count = os.getenv('VALID_GROUP_COUNT', 4)
        print(f"Valid group count threshold: {min_group_count}")
        valid_groups = valid_groups[valid_groups >= int(min_group_count)].index

        df = df[df['model'].isin(valid_groups)]
        
        return df
    
    def prepare_features(self, vins):
        """
        Convert VIN strings to sequences of integers for neural network input
        """
        if self.tokenizer is None:
            # Create tokenizer for character-level encoding
            print("Loading tokenizer from brand to have the same results in joint model")
            self.tokenizer = pickle.load(open("models/brand/vin_brand_predictor_tokenizer.pkl", "rb")) 
            self.vocab_size = len(self.tokenizer.word_index) + 1
            print(f"Vocabulary size: {self.vocab_size}")
        
        # Convert VINs to sequences
        sequences = self.tokenizer.texts_to_sequences(vins)
        # Pad sequences to ensure consistent length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return padded_sequences
    
    def prepare_labels(self, models):
        """
        Encode model labels as integers
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(models)
        else:
            encoded_labels = self.label_encoder.transform(models)
            
        return encoded_labels
    
    def build_model(self, num_classes):
        
        # Load or define your other model
        other_model = load_model('models/brand/vin_brand_predictor.keras')
        other_model.trainable = False 

        # Define input
        inputs = Input(shape=(self.max_length,))


        # Reshape to 3D: (17,) -> (17, 1)
        reshaped = Reshape((17, 1))(inputs)

        # Now Cropping1D works
        cropped = Cropping1D(cropping=(0, 6))(reshaped)  # (11, 1)

        # Flatten back to 1D
        flattened = Flatten()(cropped) 
        # Branch 1: Embedding
        embedding_output = Embedding(
            input_dim=self.vocab_size,
            output_dim=128,
            input_length=self.max_length -6
        )(flattened)

        # Branch 2: Other model (outputs shape: (None, 52))
        other_output = other_model(inputs)

        # Option 1: Repeat other_output for each position in sequence
        other_output_repeated = RepeatVector(self.max_length)(other_output)

        other_output_cropped = Cropping1D(cropping=(0, 6))(other_output_repeated)

        # Concatenate along feature dimension
        concatenated = Concatenate(axis=-1)([embedding_output, other_output_cropped])

        # Continue with rest of your model
        x = Conv1D(256, 3, activation='relu', padding='same')(concatenated)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x) 
        x = Dropout(0.5)(x)

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x) 
        x = Dropout(0.4)(x)

        outputs = Dense(num_classes, activation='softmax')(x)

        # Create final model
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_model_old(self, num_classes):
        """
        Build the neural network architecture
        """
        model = Sequential([
            # Embedding layer to convert character indices to dense vectors
            Embedding(input_dim=self.vocab_size, 
                     output_dim=64, 
                     input_length=self.max_length),
            
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalMaxPooling1D(),  # Better than Flatten
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, csv_file_path, test_size=0.2, epochs=50, batch_size=32):
        """
        Complete training pipeline
        """
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_file_path)
        
        # Prepare features and labels
        X = self.prepare_features(df['vin'].values)
        y = self.prepare_labels(df['model'].values)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        model = self.build_model(num_classes)
        print(model.summary())
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Store test data for visualization
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_classes = y_pred_classes
        self.history = history
        
        return model, history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(self.y_test, self.y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict(self, vins):
        """
        Predict model for new VIN numbers
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure VINs are in correct format
        if isinstance(vins, str):
            vins = [vins]
        
        vins = [vin.upper() for vin in vins]
        
        # Prepare features
        X = self.prepare_features(vins)
        
        # Make predictions
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert back to model names
        predicted_models = self.label_encoder.inverse_transform(predicted_classes)
        
        # Get prediction probabilities
        prediction_probs = np.max(predictions, axis=1)
        
        results = []
        for i, vin in enumerate(vins):
            results.append({
                'vin': vin,
                'Predicted_Model': predicted_models[i],
                'Confidence': prediction_probs[i]
            })
        
        return results
    
    def save_model(self, model_path="vin_model_model"):
        """
        Save the trained model and preprocessors
        """
        self.model.save(f"{model_path}.keras")
        
        # Save tokenizer and label encoder
        import pickle
        with open(f"{model_path}_tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(f"{model_path}_label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved as {model_path}")
    
    def load_model(self, model_path="vin_model_model"):
        """
        Load a trained model and preprocessors
        """
        self.model = tf.keras.models.load_model(f"{model_path}.keras")
        
        import pickle
        with open(f"{model_path}_tokenizer.pkl", 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(f"{model_path}_label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Model loaded successfully")

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = VINModelPredictor()
    
    # Train the model (replace 'your_data.csv' with your actual CSV file path)
    # The CSV should have columns 'VIN' and 'Model'
    model, history = predictor.train_model('data/brand_info_model_extracted/bim_all.csv', epochs=10)
    
    # Plot training results
    predictor.plot_training_history()
    predictor.plot_confusion_matrix()
    
    # Make predictions on new VINs
    sample_vins = ['1HGBH41JXMN109186', '2HGFG12688H509393']
    predictions = predictor.predict(sample_vins)
    
    for pred in predictions:
        print(f"VIN: {pred['vin']}")
        print(f"Predicted Model: {pred['Predicted_Model']}")
        print(f"Confidence: {pred['Confidence']:.4f}")
        print("-" * 40)
    
    # Save the model
    predictor.save_model("models/model/vin_model_predictor")