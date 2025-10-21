import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    model = load_model("models/model/vin_model_predictor.keras")
    tokenizer = pickle.load(open("models/brand/vin_brand_predictor_tokenizer.pkl", "rb")) 
    model_labels = pickle.load(open("models/model/vin_model_predictor_label_encoder.pkl", "rb"))

    vins = sys.argv[1:]

    if isinstance(vins, str):
            vins = [vins]
        
    vins = [vin.upper() for vin in vins]

    sequences = tokenizer.texts_to_sequences(vins)

    model.predictions = model.predict(np.array(sequences))

    print("Predictions:")
    for vin, prediction in zip(vins, model.predictions):
        print(f"VIN: {vin}")
        np_prediction = np.array(prediction)
        model_preds = model_labels.inverse_transform(
             np.argsort(np_prediction)[-5:][::-1]
             )


        model_confidence = np.max(np_prediction)
        print(f"Predicted Models: {model_preds}")
        print(f"Model Confidence: {model_confidence:.4f}")
        print("-" * 40)
