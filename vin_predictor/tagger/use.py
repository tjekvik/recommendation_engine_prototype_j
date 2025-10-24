import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    model = load_model("models/tagger/vin_tagger_predictor.keras")
    tokenizer = pickle.load(open("models/brand/vin_brand_predictor_tokenizer.pkl", "rb")) 
    tag_labels = pickle.load(open("models/tagger/vin_tagger_predictor_label_encoder.pkl", "rb"))

    vins = sys.argv[1:]

    if isinstance(vins, str):
            vins = [vins]
        
    vins = [vin.upper() for vin in vins]

    sequences = tokenizer.texts_to_sequences(vins)

    predictions = model.predict(np.array(sequences))
    predictions = (predictions > 0.5).astype(int)
    predictions = tag_labels.inverse_transform(predictions)

    print("Predictions:")
    for vin, prediction in zip(vins, predictions):
        print(f"VIN: {vin}")
        print(f"Predicted Tags: {prediction}")
        print("-" * 40)
