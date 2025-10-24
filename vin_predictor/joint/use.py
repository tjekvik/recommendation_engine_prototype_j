#! /usr/bin/env python3

import sys
import pickle
import tensorflow.keras as keras

import numpy as np

from tensorflow.keras.models import load_model

if __name__ == "__main__":
    model = load_model("models/joint/joint_model.keras")
    tokenizer = pickle.load(open("models/brand/vin_brand_predictor_tokenizer.pkl", "rb")) 
    brand_labels = pickle.load(open("models/brand/vin_brand_predictor_label_encoder.pkl", "rb"))
    model_labels = pickle.load(open("models/model/vin_model_predictor_label_encoder.pkl", "rb"))
    year_labels = pickle.load(open("models/year/vin_year_predictor_binarizer.pkl", "rb"))

    vins = sys.argv[1:]

    if isinstance(vins, str):
            vins = [vins]
        
    vins = [vin.upper() for vin in vins]

    sequences = tokenizer.texts_to_sequences(vins)

    model.predictions = model.predict(np.array(sequences))

    brand_end_index = len(brand_labels.classes_)
    model_end_index = brand_end_index + len(model_labels.classes_)

    print("Predictions:")
    for vin, prediction in zip(vins, model.predictions):
        print(f"VIN: {vin}")
        np_prediction = np.array(prediction)
        brand_pred = brand_labels.inverse_transform([np.argmax(np_prediction[:brand_end_index])])[0]
        model_preds = model_labels.inverse_transform(
             np.argsort(np_prediction[brand_end_index:model_end_index])[-5:][::-1]
             )

        year_tensor = np.array([np_prediction[model_end_index:]])
        year_tensor = (year_tensor > 0.5).astype(int)
        year_pred = year_labels.inverse_transform(year_tensor)[0]

        brand_confidence = np.max(np_prediction[:brand_end_index])
        model_confidence = np.max(np_prediction[brand_end_index:model_end_index])
        print(f"Predicted Models: {model_preds[0]}")
        print(f"Predicted Brand: {brand_pred}")
        print(f"Predicted Year: {year_pred}")
        print(f"Brand Confidence: {brand_confidence:.4f}")
        print(f"Model Confidence: {model_confidence:.4f}")
        print("-" * 40)
