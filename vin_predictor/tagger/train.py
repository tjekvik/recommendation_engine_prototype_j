#!/usr/bin/env python3
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from common.unifiers import unify_brand_name

class VINTaggerPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = None
        self.max_length = 17  # VIN numbers are always 17 characters
        self.vocab_size = None

    def load_and_preprocess_data(self, csv_file_path):
        """
        Load and preprocess the VIN and Brand data from CSV
        """
        # Load data
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        
        # Basic data cleaning
        df['vin'] = df['vin'].str.upper()  # Normalize VIN to uppercase
        df['info'] = df['info'].astype('str')
        
        # Validate VIN format (should be 17 characters)
        df = df[df['vin'].str.len() == 17]
        df = df[df['info'] != '']  # Remove empty info names

        print(f"After cleaning: {len(df)} records")
        
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
        
        return sequences
    
    def prepare_labels(self, infos):
        """
        Encode brand labels as integers
        """
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(min_df=50)
            self.vectorizer.fit(infos)
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit([self.vectorizer.get_feature_names_out()])
        else:
            self.vectorizer.fit(infos)
        
        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.vectorizer.get_feature_names_out()])
        encoded_labels = self.label_encoder.transform(infos.str.lower().str.split())
        return encoded_labels
    
    def build_model(self, num_classes):
        """
        Build the neural network architecture
        """
        model = Sequential([
            # Embedding layer to convert character indices to dense vectors
            Embedding(input_dim=self.vocab_size, 
                     output_dim=64, 
                     input_length=self.max_length),
            
            # Flatten the embedded sequences
            Flatten(),
            
            # Dense layers with dropout for regularization
     
            Dense(256, activation='relu'),
            Dropout(0.3),

            BatchNormalization(),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.1),

            BatchNormalization(),
            
            # Output layer
            Dense(num_classes, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
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
        y = self.prepare_labels(df['info'].astype(str))
        

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Build model
        num_classes = len(self.vectorizer.get_feature_names_out())
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.vectorizer.get_feature_names_out()}")
        
        model = self.build_model(num_classes)
        print(model.summary())

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

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
                   xticklabels=self.vectorizer.classes_,
                   yticklabels=self.vectorizer.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict(self, vins):
        """
        Predict brand for new VIN numbers
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure VINs are in correct format
        if isinstance(vins, str):
            vins = [vins]
        
        vins = [vin.upper() for vin in vins]
        
        # Prepare features
        X = self.prepare_features(vins)
        X = np.asarray(X)
        # Make predictions
        predictions = self.model.predict(X)
        predicted_binary = (predictions > 0.5).astype(int)
        predicted_labels = self.label_encoder.inverse_transform(predicted_binary)
        return predicted_labels
    
    def save_model(self, model_path="vin_year_model"):
        """
        Save the trained model and preprocessors
        """
        self.model.save(f"{model_path}.keras")
        
        # Save tokenizer and label label_encoder
        import pickle
        with open(f"{model_path}_tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(f"{model_path}label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved as {model_path}")
    
    def load_model(self, model_path="vin_year_model"):
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
    predictor = VINTaggerPredictor()
    
    # Train the model (replace 'your_data.csv' with your actual CSV file path)
    # The CSV should have columns 'VIN' and 'Brand'
    model, history = predictor.train_model('data/brand_info_model/bim_100k.csv', epochs=5)
    
    # Plot training results
    # predictor.plot_training_history()
    # predictor.plot_confusion_matrix()
    
    # Make predictions on new VINs
    sample_vins = ['1HGBH41JXMN109186', '2HGFG12688H509393']
    predictions = predictor.predict(sample_vins)
    
    for pred in predictions:
        print(pred)
    
    # Save the model
    predictor.save_model("models/tagger/vin_tagger_predictor")